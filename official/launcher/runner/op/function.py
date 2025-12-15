import torch
import torch.distributed
import os
# from launcher.utils import load_layer
from copy import deepcopy
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from omegaconf import open_dict
from .customer import get_op
from .op_swapper import OPSwapper
from .data_loader import BatchSampler, configs
from .real_data_loader import RealDataLoader, RealDataBatchSampler
from moprobe.utils import to_device, acc_check
from launcher.tools import case_loader
from moprobe.advanced_compare import draw_manager
import logging


def initialize_distributed(args):
    global configs
    world_size = int(os.environ.get('WORLD_SIZE', 0))
    rank = int(os.environ.get('LOCAL_RANK', 0))
    if world_size > 0:
        # torch.cuda.set_device(rank)
        print('world_size:', world_size, 'rank:', rank)
        torch.distributed.init_process_group(backend='gloo', world_size=world_size, rank=rank)
    configs['DEVICE'] = torch.device("cuda")
    if args.training.report:
        args.training.report += f'/{rank}'
    return configs['DEVICE'], rank

def manual_seed(rank=0, seed=42):
    torch.manual_seed(rank + seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank + seed)

def forward_loop(module, data_iterator, args):
    if data_iterator is None:
        return None
    if data_iterator.has_kwargs():
        arg, kwargs = data_iterator.get_input()
        res = module(*arg, **kwargs)
    else:
        batch, _ = data_iterator.get_input()
        res = module(*batch)
    # torch.cuda.synchronize()
    if data_iterator.out_mapping and res is None:
        res = data_iterator.get_output()
    else:
        data_iterator.check_output(res)
    if 'save_output_by_pt' in args.run and args.run.save_output_by_pt:
        draw_manager.save_pt(res, data_iterator.get_case_info())
    return res

def create_schedule(case, fast_test=False):
    run_list = []
    run_list.append({'name': case.name})

    def is_key(s):
        if any(k in s for k in ['precision', 'high', 'low']):
            return True
        if s.startswith('_'):
            return True
        return False

    if case.get('precision') and isinstance(case.precision, ListConfig):
        with open_dict(case):
            if not case.get('size'):
                case.size = DictConfig({})
            case.size.precision = case.precision

    print('case VAL:', case.size)

    target_config_path = 'fake_target_config.json'
    if os.path.exists(target_config_path):
        import json
        try:
            with open(target_config_path, 'r') as f:
                target_config = json.load(f)
                print("Successfully loaded target_config")

                # Save the original name from run_list
                name = run_list[0]['name']
                run_list.clear()

                # Get parameter combinations from all configuration types
                all_configs = []
                for config_type in target_config.values():
                    all_configs.extend(config_type)

                # Create run_list items for each config and precision
                precision_list = case.size.get('precision', ['bf16']) if isinstance(case.size.get('precision'),\
                    ListConfig) else [case.size.get('precision', 'bf16')]
                for config in all_configs:
                    for precision in precision_list:
                        item = {'name': name}
                        item.update(config)
                        if 'GS' not in item:
                            item['GS'] = 128
                        item['precision'] = precision
                        run_list.append(item)

                print("run_list generated from target_config: {}".format(run_list))
        except Exception as e:
            print("Error loading or processing target_config: {}".format(e))
            exit()
    else:
        for k, v in  case.size.items():
            len_run_list = len(run_list)
            if isinstance(v, ListConfig):
                if fast_test and not is_key(k) and len(v) > 2:
                    if fast_test == 'only_first':
                        v = [v[0]]
                    elif fast_test == 'only_last':
                        v = [v[-1]]
                    else:
                        v = [v[0], v[-1]]
                for i, _item in enumerate(v):
                    if i > 0:
                        for _i in range(len_run_list):
                            run_list.append(deepcopy(run_list[_i]))
                    for _i in range(len_run_list):
                        run_list[i * len_run_list + _i][k] = _item
            else:
                for _i in range(len_run_list):
                    run_list[_i][k] = v

    def get_val(v, _dict):
        if isinstance(v, ListConfig):
            shape = []
            for _i in v:
                shape.append(get_val(_i, _dict))
            return shape
        elif isinstance(v, DictConfig):
            shape = {}
            for key, value in v.items():
                shape[key] = get_val(value, _dict)
            return shape
        elif isinstance(v, (int , float, bool, type(None))):
            return v
        elif isinstance(v, str) and any(k in v for k in '+-*/'):
            a = eval(v, deepcopy(_dict))
            return int(a)
        elif v in _dict:
            return _dict[v]
        elif v in ['Zero', 'fp32', 'fp16', 'bf16', 'e4m3', 'e5m2', 'int32', 'int64', 'long', 'bool', 'auto', 'Int',
                   '_input_normal', '_weight_normal']:
            return v
        else:
            print('not found in process yaml:', v, _dict)
            return v

    for i, _dict in enumerate(run_list):
        data = []
        for v in case.input:
            data.append(get_val(v, _dict))
        run_list[i]['input'] = data
        for k in ['output', 'cuda_name', 'cuda_mapping', 'cpu_name', 'cpu_mapping']:
            if k in case:
                if isinstance(case[k], ListConfig):
                    run_list[i][k] = list(case[k])
                else:
                    run_list[i][k] = case[k]
        run_list[i] = DictConfig(run_list[i])
        print('schedule:', case.name, i, run_list[i])
    return run_list

def splite_test_case(args):
    if args.load_case_file:
        for i, case in enumerate(case_loader.load(args)):
            if 'schedule_case_limit' not in args.run or i < args.run.schedule_case_limit or args.run.schedule_case_limit == -1:
                yield case
        return
    for op in args.op:
        if isinstance(op, str):
            for item in args.op[op]:
                if isinstance(args.training.precision, ListConfig) and 'precision' not in item:
                    with open_dict(item):
                        item.precision = args.training.precision
                if 'size' in item or ('precision' in op and isinstance(op.precision, ListConfig)):
                    for i, case in enumerate(create_schedule(item, fast_test=args.run.fast_test)):
                        if 'schedule_case_limit' not in args.run or i < args.run.schedule_case_limit or args.run.schedule_case_limit == -1:
                            yield case
                        else:
                            break
                else:
                    yield item
        else:
            if isinstance(args.training.precision, ListConfig) and 'precision' not in op:
                with open_dict(op):
                    op.precision = args.training.precision
            if 'size' in op or ('precision' in op and isinstance(op.precision, ListConfig)):
                for i, case in enumerate(create_schedule(op, fast_test=args.run.fast_test)):
                    if 'schedule_case_limit' not in args.run or i < args.run.schedule_case_limit or args.run.schedule_case_limit == -1:
                        yield case
                    else:
                        break
            else:
                yield op

class DataLoader():
    def __init__(self):
        self.case = None
        self.dtype = None
        self.seed = 42
        self.offset = 10
        self.this_seed = 0
        self.config = None
        self.batch = None
        # 添加真实数据加载器支持
        self.real_data_loader = None

    def set_real_data_loader(self, dump_json_path: str, pt_data_dir: str, enable_real_data: bool = False):
        """
        设置真实数据加载器
        
        Args:
            dump_json_path: dump.json 文件路径
            pt_data_dir: pt 文件目录路径
            enable_real_data: 是否启用真实数据加载
        """
        if enable_real_data:
            self.real_data_loader = RealDataLoader(dump_json_path, pt_data_dir, enable_real_data)
            logging.getLogger(__name__).info(f"真实数据加载器已启用: dump_json={dump_json_path}, pt_dir={pt_data_dir}")
        else:
            self.real_data_loader = None

    def new(self, test_case, args):
        case = {
            'name': test_case.name,
            'input': test_case.input,
            'output': test_case.output if 'output' in test_case else [],
            'precision': test_case.precision if 'precision' in test_case else args.training.precision
        }
        self.case = DictConfig(case)
        self.config = DictConfig({
            'run': args.run,
            'distributions': args.task.acc
        })
        self.this_seed = self.seed
        manual_seed(seed=self.this_seed)
        
        # 使用支持真实数据的批次采样器
        if self.real_data_loader:
            self.batch = RealDataBatchSampler(self.case, self.config, self.this_seed, self.real_data_loader)
        else:
            self.batch = BatchSampler(self.case, self.config, self.this_seed)
        
        if self.batch.memory_not_match:
            print('input memory needed:', f"{self.batch.get_memory()}G,", 'will not exec')
            return None
        
        return self.batch.clone()

    def next_iter(self):
        if self.case:
            self.this_seed += self.offset
            manual_seed(seed=self.this_seed)
            
            # 使用支持真实数据的批次采样器
            if self.real_data_loader:
                self.batch = RealDataBatchSampler(self.case, self.config, self.this_seed, self.real_data_loader)
            else:
                self.batch = BatchSampler(self.case, self.config, self.this_seed)
            
            return self.batch.clone()

    def this_iter(self):
        if self.batch:
            return self.batch.clone()

data_loader = DataLoader()
draw_manager.set('get_dtype', BatchSampler.get_dtype)

def prepare_test_case(test_case, args):
    global configs

    if any(k in test_case for k in ['cuda_name', 'cpu_name']):
        Op = OPSwapper(test_case, args)
    elif test_case.name.startswith('aten'):
        Op = getattr(torch.ops.aten, test_case.name.replace('aten::', ''))
    else:
        package, op_name = test_case.name.split('::')
        Op = get_op(package, op_name)

    batch = []
    if hasattr(test_case, 'load'):
        pass
        # for case_file in test_case.load:
        #     print('case_file:', case_file)
        #     layer_name, train_iterator = load_layer(case_file, no_rank=True)
        #     print('layer_name:', layer_name)
        #     print(train_iterator)
        #     batch.append(train_iterator)
    else:
        batch = data_loader.new(test_case, args)

    if 'cuda_graph' in args.run and args.run.cuda_graph and batch:
        g = torch.cuda.CUDAGraph()
        class Layer(torch.nn.Module):
            def forward(self, *arg, **kwargs):
                return Op(*arg, **kwargs)

        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.layers = torch.nn.ModuleList([Layer() for i in range(args.run.max_steps)])

            def forward(self, *arg, **kwargs):
                for layer in self.layers:
                    x = layer(*arg, **kwargs)
                return x
        return {'cuda_graph': g, 'model': Model().to(configs['DEVICE'])}, batch, forward_loop
    elif test_case.name.startswith('apex'):
        print('not support no cuda_graph: apex, please set True for the config')

    draw_manager.set('test_case', test_case)
    return {'model': Op}, batch, forward_loop


def all_gather(obj_list, obj, fp8_transfer=False):
    torch.distributed.all_gather_object(obj_list, obj=obj)


def run_iter(module, data_iterator, forward_backward_func, num_steps, args):
    global configs
    res = []
    if 'cuda_graph' in args.run and args.run.cuda_graph:
        forward_backward_func(module, data_iterator, args)
    else:
        word_size  = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        for _ in range(int(num_steps/word_size)):
            if 'acc_check' in args.run and args.run.acc_check:
                assert (not num_steps%word_size)
                assert (word_size < 3)
                if not data_iterator:
                    data_iterator = data_loader.next_iter()
                if torch.distributed.is_initialized():
                    rank = torch.distributed.get_rank()
                    device = str(configs['DEVICE']).replace('cuda', 'device') if 'cuda' in str(configs['DEVICE']) else str(configs['DEVICE']).replace('device', 'cuda')
                    data_iterators = [None, None]
                    all_gather(data_iterators, obj=data_iterator, fp8_transfer=True)
                    outs = []
                    for data_iterator in data_iterators:
                        data_iterator.to(torch.device(configs['DEVICE']))
                        outs.append(forward_backward_func(module, data_iterator, args))
                    out_cpu = to_device(torch.device(args.run.acc_target_device), outs[1 - rank], fp8_transfer=True)
                    targets = [None, None]
                    all_gather(targets, obj=out_cpu)
                    out = outs[rank]
                    target = targets[1-rank]
                    if args.run.tripartite_check:
                        data_iterator = data_loader.this_iter().to(args.run.acc_target_device, args.run.use_cpu_hight_precision)
                        tripartite_target = forward_backward_func(module, data_iterator, args)
                else:
                    # all_gather(None, obj=data_iterator, fp8_transfer=True)
                    device = args.run.acc_target_device
                    out = forward_backward_func(module, data_iterator, args)
                    data_iterator = data_loader.this_iter().to(device, args.run.use_cpu_hight_precision)
                    target = forward_backward_func(module, data_iterator, args)
                api_name = module.__name__
                del data_iterator
                data_iterator = None
                torch.cuda.empty_cache()
                res.extend(acc_check(api_name, device, target, out, args.run.use_allclose, args.run.use_detail_check))
                if args.run.tripartite_check and torch.distributed.is_initialized():
                    res.extend(acc_check(api_name, args.run.acc_target_device, tripartite_target, out, args.run.use_allclose, args.run.use_detail_check))
            else:
                forward_backward_func(module, data_iterator, args)
    return res

def get_report_format(args):
    if 'acc_check' in args.run and args.run.acc_check:
        if args.run.use_detail_check:
            return ['status', 'b_device', 'b_dtype', 'rtol', 'atol', 'cosine_sim', 'max_abs_err', 'rel_err_hundredth', 'rel_err_thousandth', 'rel_err_ten_thousandth', 'error_rate', 'eb', 'rmse', 'small_value_err_ratio', 'max_rel_error', 'mean_rel_error', 'max_ulp_error', 'mean_ulp_error', 'ulp_error_proportion', 'message']
        else:
            return ['status', 'b_device', 'b_dtype', 'rtol', 'atol', 'max_abs_err', 'max_relative_err', 'ten_thousand_err', 'one_thousand_err', 'five_thousand_err', 'hundred_err']
    else:
        return None
