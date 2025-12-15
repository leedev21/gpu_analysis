import os
from importlib import import_module
import torch
from torch.optim import Adam
from torch.utils.data import Dataset
from moprobe.utils import load_to_device, to_device
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
try:
    from torchtrace.torchtrace import update
except ImportError:
    def update(*args, **kwargs):
        pass

configs = {
    'DEVICE': None,
    'IntTensor_zero': None
}

def initialize_distributed(args):
    global configs

    rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', 0))
    if rank == -1 or world_size == 0:
        configs['DEVICE'] = torch.device("cuda")
        return configs['DEVICE'], 0

    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    print('world_size:', world_size, 'rank:', rank)

    try:
        import vllm
        from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
        init_distributed_environment(world_size=world_size, rank=rank, distributed_init_method="env://", local_rank=0, backend="nccl")
        tensor_model_parallel_size = 1
        pipeline_model_parallel_size = 1
        context_parallel_size = 1
    except:
        try:
            from megatron.core import parallel_state
            tensor_model_parallel_size = args.training.tensor_model_parallel_size
            pipeline_model_parallel_size = args.training.pipeline_model_parallel_size
            context_parallel_size = args.training.context_parallel_size
            parallel_state.destroy_model_parallel()

            torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)

            # Megatron core distributed training initialization
            parallel_state.initialize_model_parallel(tensor_model_parallel_size,
                                                    pipeline_model_parallel_size,
                                                    context_parallel_size=context_parallel_size)
        except:
            raise Exception('distributed training initialization failed')
    torch.cuda.set_device(rank)
    configs['DEVICE'] = torch.device("cuda")
    print('DEVICE:', configs['DEVICE'], 'tp:', tensor_model_parallel_size,
          'pp:', pipeline_model_parallel_size, 'cp:', context_parallel_size)
    return configs['DEVICE'], rank


def manual_seed(rank=0, seed=42):
    torch.manual_seed(rank + seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank + seed)
    if torch.distributed.is_initialized():
        try:
            from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
            model_parallel_cuda_manual_seed(42)
        except:
            pass


def get_cls(path, module_name):
    module = import_module(path)
    if module and hasattr(module, module_name):
        return getattr(module, module_name)
    return None


class RandomDataset(Dataset):
    def __init__(self, shape_list, n_samples, dtype=torch.half):
        self.len = n_samples
        self.data = []
        self.type = 'randn'
        for shape in shape_list:
            shape = list(shape)
            print(shape)
            if isinstance(shape[0], str):
                if shape[0] == 'emb':
                    random_floats = torch.rand(n_samples, shape[1], 1) * shape[2]
                    _tensor = random_floats.reshape(n_samples, -1).to(torch.int64)
            else:
                shape.insert(0, n_samples)
                _tensor = torch.randn(shape, dtype=dtype)
            self.data.append(_tensor)
        ### need update

    def __getitem__(self, index):
        items = [self.data[k][index] for k in range(len(self.data))]
        return items

    def __len__(self):
        return self.len


def vanilla_loop(module, batch, optim):
    print('vanilla_loop')
    loss = module(*batch)
    loss.backward()
    print(loss)
    optim.step()
    optim.zero_grad()
    print('optim done')


def forward_loop(module, batch, optim):
    if isinstance(batch[-1], dict):
        arg = batch[:-1]
        kwargs = batch[-1]
        return module(*arg, **kwargs)
    else:
        return module(*batch)


def get_dtype(args):
    if args.module.layer.precision in [32, 'fp32']:
        return torch.float
    if args.module.layer.precision in [16, 'fp16']:
        return torch.half
    if args.module.layer.precision in ['bf16']:
        return torch.bfloat16
    return None


def splite_test_case(args):
    layer_name = args.module.layer.name
    module_path = 'runner.module.layers.'
    get_module_func = get_cls(module_path + layer_name, 'get_module')
    module_splite_test_case = get_cls(module_path + layer_name, 'splite_test_case')
    env_func = get_cls(module_path + layer_name, 'env')
    data_loader_func = get_cls(module_path + layer_name, 'data_loader')
    for test_case in module_splite_test_case(args):
        test_case['_get_module'] = get_module_func
        test_case['_env'] = env_func
        test_case['_dataloader'] = data_loader_func
        yield test_case


def prepare_test_case(test_case, args):
    dtype = get_dtype(args)
    n_samples = args.run.loop_time
    test_case['_env'](test_case['type'])
    args_init = test_case['init'] if test_case.get('init') else {}
    args_init['precision'] = args.module.layer.precision
    args_init['forward_only'] = args.training.forward_only
    args_init['distribution'] = args.task.acc._weight_normal
    args_init['load_case_file'] = args.load_case_file
    args_init['save_output_by_pt'] = args.run.save_output_by_pt if 'save_output_by_pt' in args.run else False

    device = torch.device(configs['DEVICE']) if 'init_cpu' not in test_case['init']['config'] else torch.device('cpu')

    module = test_case['_get_module'](test_case['type'], args_init).to(device)
    shape_list = [test_case['input']] if 'seq_length' in test_case['input'] else [test_case['input']]
    # shape_list = args.module.layer.input
    for i in range(len(shape_list)):
        if isinstance(shape_list[i], (list, tuple, ListConfig)):
            for k in range(len(shape_list[i])):
                if shape_list[i][k] == 'seq_length':
                    shape_list[i][k] = test_case['seq_length']
    dataset = test_case['_dataloader'](shape_list, n_samples, dtype, args=args)
    dataset.to(device)
    # batch = [x.to(configs['DEVICE']) for x in dataset[0]]
    # batch = [to_device(device, x) for x in dataset]

    _loop = forward_loop if args.training.forward_only else vanilla_loop
    optim = None if args.training.forward_only else Adam(module.parameters())
    return {'model': (module, optim)}, dataset, _loop


def run_iter(model_optim, data_iterator, forward_backward_func, num_steps, args):
    module, optim = model_optim
    outs = []

    for i in range(num_steps):
        if update and os.getenv('RUN_TYPE', '') != 'test':
            update('step', 'start', i)
        data = data_iterator.next_iter()
        if data is None:
            break
        if 'acc_check' in args.run and args.run.acc_check:
            print(i, len(data))
            outs.extend(forward_backward_func(module, data, optim))
        else:
            forward_backward_func(module, data, optim)
        if update and os.getenv('RUN_TYPE', '') != 'test':
            update('step', 'end', i)
    return outs


def get_report_format(args):
    if 'acc_check' in args.run and args.run.acc_check:
        if args.run.use_detail_check:
            return ['status', 'b_device', 'b_dtype', 'rtol', 'atol', 'cosine_sim', 'max_abs_err', 'rel_err_hundredth', 'rel_err_thousandth', 'rel_err_ten_thousandth', 'error_rate', 'eb', 'rmse', 'small_value_err_ratio', 'max_rel_error', 'mean_rel_error', 'max_ulp_error', 'mean_ulp_error', 'ulp_error_proportion', 'message']
        else:
            return ['status', 'b_device', 'b_dtype', 'rtol', 'atol', 'max_abs_err', 'max_relative_err', 'ten_thousand_err', 'one_thousand_err', 'five_thousand_err', 'hundred_err']
    else:
        return None