import os
import sys
import torch
from importlib import import_module
import time
from torchtrace.utils import GLOBAL_CFG
from torchtrace.module import tracker
import copy
import inspect
from functools import partial
import subprocess
from torch.distributed.rendezvous import register_rendezvous_handler, rendezvous  # noqa: F401
from torch._C._distributed_c10d import PrefixStore, Store, ProcessGroup
import contextlib
from torchtrace.torch_dispatch import myTorchDispatchMode


from torchtrace.utils import (
    parser_map,
    class_name,
    default_parser,
    get_device_id,
    read_file,
    GLOBAL_CFG,
    get_save_path,
    print_rank_0
)
from torchtrace.library import _load_transformer_engine_torch, _load_apex_torch, _load_vllm_triton, _load_jit_library

LOGGING_PATH = 'results/log'

def logging_set(name=None):
    global LOGGING_PATH
    if name:
        LOGGING_PATH = os.path.join('results/log', name)
    if not os.path.isdir(LOGGING_PATH):
        os.makedirs(LOGGING_PATH, exist_ok=True)

layer_dict = {}

def get_cls(path, module_name):
    module = import_module(path)
    if module and hasattr(module, module_name):
        return getattr(module, module_name)
    return None

def layer_iter_forward(module_name, next=False):
    global layer_dict
    if module_name not in layer_dict:
        layer_dict[module_name] = {'n': 0, 'b': 0}
    elif next:
        layer_dict[module_name]['n'] += 1
        layer_dict[module_name]['b'] = layer_dict[module_name]['n']
    layer_iter = layer_dict[module_name]['b']
    return layer_iter

def layer_iter_backward(module_name, next=False):
    global layer_dict
    layer_iter = layer_dict[module_name]['b']
    if module_name not in layer_dict:
        layer_dict[module_name] = {'n': 0, 'b': 0}
    elif next:
        layer_dict[module_name]['b'] -= 1
    return layer_iter

# 定义前向传播中的 hook
def forward_hook_fn(name, conf):
    def hook_fn(module, args, kwargs, module_out):
        if conf['root'] or conf['layer']:
            torch.cuda.nvtx.range_pop()
        if conf['nvtx']:
            return
        module_name = module.__class__.__name__
        iter = layer_iter_forward(module_name)
        tracker.trace_module('fwd', 'return', iter, module_name, name, module, args, kwargs, module_out, get_data)
    return hook_fn

# 定义反向传播中的 hook
def backward_hook_fn(name, conf):
    def hook_fn(module, module_in, module_out):
        if GLOBAL_CFG['sync_mode']:
            torch.cuda.synchronize()
        if conf['layer']:
            torch.cuda.nvtx.range_pop()
        if conf['nvtx'] or conf['root']:
            return
        module_name = module.__class__.__name__
        iter = layer_iter_backward(module_name, next=True)
        tracker.trace_module('bwd', 'return', iter, module_name, name, module, module_in, None, module_out, get_data)
    return hook_fn

# 定义前向传播中的 pre hook
def forward_pre_hook_fn(name, conf):
    def hook_fn(module, args, kwargs):
        if conf['root']:
            torch.cuda.nvtx.range_push("trace::fwd:model")
        if conf['layer']:
            torch.cuda.nvtx.range_push(f"trace::fwd:{name}")
        if conf['nvtx']:
            return
        module_name = module.__class__.__name__
        iter = layer_iter_forward(module_name, next=True)
        tracker.trace_module('fwd', 'call', iter, module_name, name, module, args, kwargs, None, get_data, conf['root'])
    return hook_fn

# 定义反向传播中的 pre hook
def backward_pre_hook_fn(name, conf):
    def hook_fn(module, grad_output):
        if GLOBAL_CFG['sync_mode']:
            torch.cuda.synchronize()
        # if conf['root']:
        #     torch.cuda.nvtx.range_push("trace::bwd:model")
        if conf['layer']:
            torch.cuda.nvtx.range_push(f"trace::bwd:{name}")
        if conf['nvtx'] or conf['root']:
            return
        module_name = module.__class__.__name__
        iter = layer_iter_backward(module_name)
        tracker.trace_module('bwd', 'call', iter, module_name, name, module, grad_output, None, None, get_data)
    return hook_fn

# 定义优化器 optim step 的 pre hook
def optim_pre_hook_fn(name, conf):
    def hook_fn(optimizer, args, kwargs):
        if conf['nvtx']:
            torch.cuda.nvtx.range_push("trace::optim:step")
            return
        tracker.trace_optimizer(name, optimizer.__class__.__name__, 'Pre OptimStep')
    return hook_fn

# 定义优化器 optim step 的 post hook
def optim_post_hook_fn(name, conf):
    def hook_fn(optimizer, args, kwargs):
        if conf['nvtx']:
            torch.cuda.nvtx.range_pop()
            return
        tracker.trace_optimizer(name, optimizer.__class__.__name__, 'OptimStep')
    return hook_fn

# 定义优化器 optim step 的 hook
def optim_hook_fn(name, step, conf):
    def hook_fn(self):
        if conf['nvtx']:
            torch.cuda.nvtx.range_push("trace::optim:step")
        else:
            tracker.trace_optimizer('optimizer', name, 'pre_optim:step')
        outs = step(self)
        if conf['nvtx']:
            torch.cuda.nvtx.range_pop()
        else:
            tracker.trace_optimizer(name, name, 'post_optim:step')
        return outs
    return hook_fn

# 定义saved tensor forwards 的 pack_hook
def pack_hook_fn(name=None):
    def hook_fn(x):
        tracker.print_msg(get_shape(x), 'step', x.grad_fn)
    return hook_fn

# 定义saved tensor backwards 的 unpack_hook
def unpack_hook_fn(name=None):
    def hook_fn(x):
        tracker.print_msg(get_shape(x), 'step', x.grad_fn)
    return hook_fn

# 获取function input
def get_func_input(*args, **kwargs):
    device_id = get_device_id(args) if args else None
    saved_args = []
    saved_kwargs = {}
    args_str = ""
    kwargs_str = ", "
    for i, tensor in enumerate(args):
        parser = parser_map.get(class_name(tensor),
                                default_parser)
        trace_value, trace_str = parser(tensor, save=False)
        saved_args.append(trace_str)
        args_str += f"{trace_value.getString()}, "
    args_str = args_str[:-2]
    saved_kwargs = {}
    for key, tensor in kwargs.items():
        parser = parser_map.get(class_name(tensor),
                                default_parser)
        trace_value, trace_str = parser(tensor, save=False)
        saved_kwargs[key] = trace_str
        kwargs_str += f"{key}={trace_value.getString()}, "
    kwargs_str = kwargs_str[:-2]
    return device_id, args_str, kwargs_str, saved_args, saved_kwargs

# 定义function 的 hook
def function_hook_fn(name, func, src_file):
    def hook_fn(*args, **kwargs):
        device_id = get_device_id(args) if args else None
        cache = tracker.trace_api(name, 'call', device_id, src_file, args, kwargs)
        outs = func(*args, **kwargs)
        device_id_out = get_device_id(outs)
        if device_id_out:
            device_id = device_id_out
        tracker.trace_api(name, 'return', device_id, src_file, outs=outs, cache=cache)
        return outs
    return hook_fn

def customer_op_hook_fn(op_type, op_name, func, src_file):
    def hook_fn(*args, **kwargs):
        name = op_type + op_name
        need_skip, need_save = myTorchDispatchMode.op_filter(name)
        if need_skip:
            return func(*args, **kwargs)
        device_id = get_device_id(args) if args else None
        if GLOBAL_CFG.get('fallback_gpu') and name == 'flash_attn::varlen_fwd':
            tracker.start_fallback_gpu()
            myTorchDispatchMode.skip = True
            tracker.set_fallback_gpu_debug(func)
        cache = tracker.trace_op(name, 'call',  device_id, need_save, args, kwargs)
        if GLOBAL_CFG.get('fallback_gpu') and name == 'flash_attn::varlen_fwd':
            myTorchDispatchMode.skip = False
            if len(cache) == 5:
                fallback_result = cache[3]
                fallback_flag = cache[4]
                cache = cache[:3]
                return fallback_result
        outs = func(*args, **kwargs)
        device_id_out = get_device_id(outs)
        if device_id_out:
            device_id = device_id_out
        tracker.trace_op(name, 'return', device_id, need_save, outs=outs, cache=cache)
        return outs
    return hook_fn

def triton_op_hook_fn(op_type, op_name):
    def hook_fn(*args, **kwargs):
        name = op_type + op_name
        need_skip, need_save = myTorchDispatchMode.op_filter(name)
        if need_skip:
            return
        device_id = get_device_id(args) if args else None
        tracker.trace_op(name, 'return',  device_id, need_save, args, kwargs) # user return interface for saved
    return hook_fn

hooks = []

def trace_module(model, name, root=False):
    conf = {'root': root, 'nvtx': GLOBAL_CFG['nvtx'], 'layer': False}
    module_name = model.__class__.__name__
    if module_name not in ["IdentityFuncOp", "IdentityOp", "Identity", "FusedScaleMaskSoftmax"]:
        if conf['nvtx']:
            conf['layer'] = any(l in module_name for l in GLOBAL_CFG['nvtx_args']['layer'])
            if not root and not conf['layer']:
                return None
        print_rank_0('hook -->', name, '<==>', module_name)
        import time
        f_name = '{}_{}_{}_rank{}'.format(name, module_name, 'init', torch.distributed.get_rank())
        print_rank_0("hook_f_name--------------------------------", f_name)
        if GLOBAL_CFG.get('module_trace') and (name in GLOBAL_CFG['module_trace']['module']
                                               or module_name in GLOBAL_CFG['module_trace']['cls']):
            import copy
            dict_bak = {}
            for k, v in model.__dict__.items():
                if any(a in k for a in ["ep_group", "batched_router_logits", "_modules", "hook", "kv_cache", "quant_method", "impl", "_forward_method"]):
                    continue
                if k == '_parameters':
                    if 'FusedMoE' in module_name or "Attention" in module_name:
                        continue
                    dict_bak[k] = {i: j.detach().cpu() for i, j in model.state_dict().items()}
                else:
                    dict_bak[k] = copy.deepcopy(v)

            res = {
                'info': {'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), 'f_name': f_name},
                'init': {'name': name, 'model': dict_bak},
                'inputs': None,
                'inputs_str': None,
                'outputs': None,
                'outputs_str': None
            }
            save_pt('init', res)
        forward_pre_hook_handle = model.register_forward_pre_hook(forward_pre_hook_fn(name, conf), with_kwargs=True)
        forward_hook_handle = model.register_forward_hook(forward_hook_fn(name, conf), with_kwargs=True)
        backward_pre_hook_handle = model.register_full_backward_pre_hook(backward_pre_hook_fn(name, conf))
        backward_full_hook_handle = model.register_full_backward_hook(backward_hook_fn(name, conf))
        hooks.append((forward_pre_hook_handle, forward_hook_handle, backward_pre_hook_handle, backward_full_hook_handle))
    return None

def search_modules(layer_name):
    def search(model, name):
        if name == layer_name:
            return model
    return search

def load_module(model, name, root=False):
    print_rank_0('-->', name, '<==>', model)
    if any(k == name for k in ['GPT.decoder', 'GPT.output_layer']):
        return None
    if model.__class__.__name__ not in ["IdentityFuncOp", "IdentityOp"]:
        forward_hook = forward_pre_hook_fn(name)
        backward_hook = backward_pre_hook_fn(name)
        forward_hook_handle = model.register_forward_pre_hook(forward_hook, with_kwargs=True)
        backward_hook_handle = model.register_full_backward_pre_hook(backward_hook)
        hooks.append((forward_hook_handle, backward_hook_handle))
    return None

def all_modules(model, names, run, root=True):
    tracker.new_model(names, model)
    if root:
        res = run(model=model, name=names, root=True)
    for name, layer in model.named_children():
        name = names + "." + name
        res = all_modules(layer, name, run, root=False)
        if res:
            return res
        res = run(model=layer, name=name)
        if res:
            return res
    # torch.autograd.graph.saved_tensors_hooks(pack_hook_fn(), unpack_hook_fn())
    return None

def trace_optm(optim, state=''):
    if GLOBAL_CFG.get('perf', False):
        return None
    if GLOBAL_CFG['nvtx']:
        if 'start' in state:
            torch.cuda.nvtx.range_push("trace::optim:step")
        elif 'end' in state:
            torch.cuda.nvtx.range_pop()
        else:
            print('Err state:', state)
            exit()
    else:
        tracker.trace_optimizer('optimizer', optim.__class__.__name__, state)

    # print('hook -->', 'optim', '<==>', optim.__class__.__name__)
    # optim_step = copy.deepcopy(optim.step)
    # optim.step = partial(optim_hook_fn(optim.__class__.__name__, optim_step))
    # optim_pre_hook_handle = optim.register_step_pre_hook(optim_pre_hook_fn('Optimizer'))
    # optim_post_hook_handle = optim.register_step_post_hook(optim_post_hook_fn('Optimizer'))
    # hooks.append((optim_pre_hook_handle, optim_post_hook_handle))


def remove_hook():
    # 移除所有 hooks
    for items in hooks:
        for hook in items:
            hook.remove()

def save_pt(stage, data):
    if data:
        path = os.path.join(get_save_path(), data['info']['f_name'] + '_trace.pt')
        if stage == 'bwd':
            res = torch.load(path, map_location=torch.device('cpu'))
            if GLOBAL_CFG['moduel_save_state']:
                res['state_bwd'] = data['state_dict']
            res['out_bwd'] = data['inputs']
            res['in_bwd'] = data['outputs']
            torch.save(res, path)
        if stage == 'init' and GLOBAL_CFG.get('init_without_params', True):
            save_path = get_save_path()
            root_path = os.path.dirname(os.path.dirname(save_path))
            # 如果上两级目录不存在，使用 base_path 作为根目录
            if not os.path.exists(root_path):
                root_path = save_path
                print(f"Upper directories do not exist. Using save path as root directory: {root_path}")
            # 创建 init 目录
            init_dir = os.path.join(root_path, 'init')
            if not os.path.exists(init_dir):
                os.makedirs(init_dir)
                print(f"Created directory: {init_dir}")

            # 构造保存路径
            path = os.path.join(init_dir, data['info']['f_name'] + '_trace.pt')
            # 保存数据
            torch.save(data, path)
            print(f"Data saved to {path}")
        else:
            torch.save(data, path)

def value_type(a):
    if a is None:
        return 'None'
    if isinstance(a, torch.Tensor):
        return 'Tensor'
    if isinstance(a, (list, tuple)):
        return 'List'
    if isinstance(a, dict):
        return 'Dict'
    if isinstance(a, (int, float, str)):
        return 'Value'
    return 'Others'

def get_data(name, f_name, module, module_out, args, kwargs=None):
    def weight_read(module):
        i = 0
        data = {}
        for name, para in module.named_parameters():
            if value_type(para) == 'Tensor' and para.requires_grad:
                # print(str(i) + ':' + name)
                data[name] = para.cpu()
                i += 1
        return data

    def args_read(data):
        saved_args = []
        if data is None:
            return data, saved_args
        if value_type(data) == 'List':
            for i, tensor in enumerate(data):
                parser = parser_map.get(class_name(tensor),
                                        default_parser)
                trace_value, trace_str = parser(tensor, save=False)
                saved_args.append(trace_str)
        if "LogitsProcessor" in f_name and not isinstance(data, torch.Tensor):
            return data[1:], saved_args
        return data, saved_args

    def kwargs_read(data):
        saved_kwargs = {}
        for key, tensor in data.items():
            parser = parser_map.get(class_name(tensor),
                                    default_parser)
            trace_value, trace_str = parser(tensor, save=False)
            saved_kwargs[key] = trace_str
        return data, saved_kwargs

    if kwargs is None:
        inputs, inputs_str = args_read(args)
    else:
        inputs, inputs_str = zip(args_read(args), kwargs_read(kwargs))
    outputs, outputs_str = args_read(module_out)

    res = {
        'info': {'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                 'f_name': f_name},
        'init': {'name': name},
        'inputs': inputs,
        'inputs_str': inputs_str,
        'outputs': outputs,
        'outputs_str': outputs_str
    }
    if GLOBAL_CFG['moduel_save_state']:
        res['state_dict'] = weight_read(module)
    return res

def each_file(dir, endwith='nsys-rep', with_path=False):
    paths = os.walk(dir)
    for path, dir_lst, file_lst in paths:
        for file_name in file_lst:
            if file_name.endswith(endwith):
                if with_path:
                    yield path, os.path.join(path, file_name)
                else:
                    yield os.path.join(path, file_name)

def get_shape(data):
    if data is None:
        return 0
    elif isinstance(data, dict):
        return {k: get_shape(v) for k, v in data.items()}
    elif isinstance(data, (tuple, list)):
        return [get_shape(v) for v in data]
    elif isinstance(data, torch.Tensor):
        return list(data.shape)
    elif isinstance(data, torch.distributed.distributed_c10d.ProcessGroup):
        return torch.distributed.get_process_group_ranks(data)
    else:
        try:
            _dict = dict(data)
            return {k: get_shape(v) for k, v in data.items()}
        except:
            return data.__class__.__name__

def get_line(lines, sep=', ', b='', e=']'):
    str_list = []
    for item in lines:
        if isinstance(lines , dict):
            str_list.append(get_line((item, lines[item]), ': ', '', ''))
        elif isinstance(item , str):
            str_list.append(item)
        elif isinstance(item , (int, float, type(None))):
            str_list.append('{} <{}>'.format(item, type(item)))
        elif isinstance(item, torch.Tensor):
            str_list.append('{} <{}, {}, {}>'.format(item.shape, item.dtype, item.device, item.grad_fn))
        elif isinstance(item, (list, tuple)):
            str_list.append(get_line(item, ', ', '[', ']'))
        elif isinstance(item, dict):
            str_list.append(get_line(item, '; ', '{', '}'))
    return b + sep.join(str_list) + e

def get_value_str(data):
    if isinstance(data, list) or isinstance(data, tuple):
        res = []
        for item in data:
            res.append(get_value_str(item))
        return res
    if inspect.isclass(data):
        data = str(data).split("'")[1]
    elif inspect.isgenerator(data):
        data = 'isgenerator'
    else:
        try:
            data = str(data)
        except:
            if inspect.isclass(type(data)):
                data = str(type(data)).split("'")[1]
            else:
                data = str(type(data))
    return data

def get_shape_to_list(data):
    shape = get_shape(data)
    if isinstance(shape, tuple):
        shape = [shape]
    return shape

def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()

def save_log(line, *args, **kwargs):
    t = 'a'
    if t in kwargs:
        t = kwargs['t']
    rank = get_rank()
    if rank != -1:
        path = 'process' + str(rank) + '.log'
    else:
        path = 'process.log'
    if len(args) > 0:
        line = [line]
        line.extend(list(args))
    if not isinstance(line , str):
        line = get_line(line)
    with open(os.path.join(LOGGING_PATH, path), t) as f:
        f.writelines(line + '\n')

def save_log_tensor_shape(tag, typ, key=None):
    if key:
        save_log('\t\t' + tag + ': ' + key + ' - ' + str(typ))
    else:
        save_log('\t\t' + tag + ': ' + str(typ))

def register_comm_hook(trace_comm):
    print_rank_0("=============COMM_HOOK Open=============")
    class C10Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor, tag):
            return tensor

        @staticmethod
        def symbolic(g, tensor, tag):
            return g.op('custom::' + tag, tensor)

        def save_log(op_name, data, train_state, npz_num):
            frame = inspect.currentframe().f_back.f_back.f_back
            local = frame.f_locals
            if 'self' in local:
                save_log('layer: ' + get_value_str(local['self']).split('\n')[0].strip('()'))
            elif 'ctx' in local:
                save_log('layer: ' + str(local['ctx']))
            else:
                save_log('layer: ' + str(frame))
            op_name = op_name + '_N' + str(10000 + npz_num)
            save_log('\t' + train_state + ': ' + op_name)
            shape_list = []
            for tensor_name, tensor in data.items():
                if isinstance(tensor, str):
                    tensor_shape = tensor
                else:
                    tensor_shape = get_shape_to_list(tensor)
                save_log_tensor_shape(tensor_name, tensor_shape)
                shape_list.append((tensor_name, tensor_shape))
            # memory = torch.cuda.memory_allocated() / 1024 / 1024
            # save_log_tensor_shape('memory', memory)
            return op_name, shape_list #, memory

    torch_all_reduce = copy.deepcopy(torch.distributed.all_reduce)
    torch_reduce_scatter_base = copy.deepcopy(torch.distributed._reduce_scatter_base)
    torch_reduce_scatter = copy.deepcopy(torch.distributed.reduce_scatter)
    torch_reduce_scatter_tensor = copy.deepcopy(torch.distributed.reduce_scatter_tensor)
    torch_all_gather_base = copy.deepcopy(torch.distributed._all_gather_base)
    torch_all_gather = copy.deepcopy(torch.distributed.all_gather)
    torch_all_gather_into_tensor = copy.deepcopy(torch.distributed.all_gather_into_tensor)
    torch_all_gather_object = copy.deepcopy(torch.distributed.all_gather_object)
    torch_all_to_all = copy.deepcopy(torch.distributed.all_to_all)
    torch_all_to_all_single = copy.deepcopy(torch.distributed.all_to_all_single)
    torch_broadcast = copy.deepcopy(torch.distributed.broadcast)
    torch_isend = copy.deepcopy(torch.distributed.isend)
    torch_irecv = copy.deepcopy(torch.distributed.irecv)
    torch_batch_isend_irecv = copy.deepcopy(torch.distributed.batch_isend_irecv)
    torch_P2POp = copy.deepcopy(torch.distributed.P2POp)
    torch_barrier = copy.deepcopy(torch.distributed.barrier)
    torch_get_world_size = copy.deepcopy(torch.distributed.get_world_size)
    torch_get_rank = copy.deepcopy(torch.distributed.get_rank)
    torch_is_initialized = copy.deepcopy(torch.distributed.is_initialized)
    torch_get_process_group_ranks = copy.deepcopy(torch.distributed.get_process_group_ranks)
    torch_get_global_rank = copy.deepcopy(torch.distributed.get_global_rank)
    torch_new_group = copy.deepcopy(torch.distributed.new_group)


    def local_all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        cmd = 'ALL_reduce'
        tracker.trace_comm(C10Function, cmd, {'op': str(op), 'out': tensor, 'group': group})
        if GLOBAL_CFG['c10_by_pass'] or torch_get_world_size(group=group) == GLOBAL_CFG.get('dummy_data_parallel_size', -1):
            if async_op:
                return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
            else:
                return
        return torch_all_reduce(tensor, op=op, group=group, async_op=async_op)

    def local_reduce_scatter_base(output, input, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        cmd = 'Reduce_scatter'
        tracker.trace_comm(C10Function, cmd, {'op': str(op), 'in': input, 'out': output, 'group': group})
        if GLOBAL_CFG['c10_by_pass'] or torch_get_world_size(group=group) == GLOBAL_CFG.get('dummy_data_parallel_size', -1):
            if GLOBAL_CFG['c10_vit_real_tensor']:
                output.copy_(input[:output.shape[0]])
            if async_op:
                return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
            else:
                return
        return torch_reduce_scatter_base(output, input, op=op, group=group, async_op=async_op)

    def local_reduce_scatter(output, input_list, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        cmd = 'Reduce_scatter'
        tracker.trace_comm(C10Function, cmd, {'op': str(op), 'in': input_list, 'out': output, 'group': group})
        if GLOBAL_CFG['c10_by_pass'] or torch_get_world_size(group=group) == GLOBAL_CFG.get('dummy_data_parallel_size', -1):
            if GLOBAL_CFG['c10_vit_real_tensor']:
                output.copy_(input_list[0])
            if async_op:
                return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
            else:
                return
        return torch_reduce_scatter(output, input_list, op=op, group=group, async_op=async_op)

    def local_reduce_scatter_tensor(output, input, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
        cmd = 'Reduce_scatter'
        tracker.trace_comm(C10Function, cmd, {'op': str(op), 'in': input, 'out': output, 'group': group})
        if GLOBAL_CFG['c10_by_pass'] or torch_get_world_size(group=group) == GLOBAL_CFG.get('dummy_data_parallel_size', -1):
            if GLOBAL_CFG['c10_vit_real_tensor']:
                output.copy_(input[:output.shape[0]])
            if async_op:
                return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
            else:
                return
        return torch_reduce_scatter_tensor(output, input, op=op, group=group, async_op=async_op)

    def local_all_gather_base(output_tensor, input_tensor, group=None, async_op=False):
        cmd = 'ALL_gather'
        tracker.trace_comm(C10Function, cmd, {'in': input_tensor, 'out': output_tensor, 'group': group})
        if GLOBAL_CFG['c10_by_pass'] or torch_get_world_size(group=group) == GLOBAL_CFG.get('dummy_data_parallel_size', -1):
            if GLOBAL_CFG['c10_vit_real_tensor']:
                target_shape = [1 for i in output_tensor.shape]
                target_shape[0] = int(output_tensor.shape[0] / input_tensor.shape[0])
                output_tensor.copy_(input_tensor.repeat(target_shape))
            if async_op:
                return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
            else:
                return
        return torch_all_gather_base(output_tensor, input_tensor, group=group, async_op=async_op)

    def local_all_gather(tensor_list, tensor, group=None, async_op=False):
        cmd = 'ALL_gather'
        tracker.trace_comm(C10Function, cmd, {'in': tensor, 'out': tensor_list, 'group': group})
        if GLOBAL_CFG['c10_by_pass'] or torch_get_world_size(group=group) == GLOBAL_CFG.get('dummy_data_parallel_size', -1):
            if GLOBAL_CFG['c10_vit_real_tensor']:
                for i in range(len(tensor_list)):
                    tensor_list[i].copy_(tensor)
            if async_op:
                return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
            else:
                return
        return torch_all_gather(tensor_list, tensor, group=group, async_op=async_op)

    def local_all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
        cmd = 'ALL_gather'
        tracker.trace_comm(C10Function, cmd, {'in': input_tensor, 'out': output_tensor, 'group': group})
        if GLOBAL_CFG['c10_by_pass'] or torch_get_world_size(group=group) == GLOBAL_CFG.get('dummy_data_parallel_size', -1):
            if GLOBAL_CFG['c10_vit_real_tensor']:
                target_shape = [1 for i in output_tensor.shape]
                target_shape[0] = int(output_tensor.shape[0] / input_tensor.shape[0])
                output_tensor.copy_(input_tensor.repeat(target_shape))
            if async_op:
                return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
            else:
                return
        return torch_all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op)

    def local_all_gather_object(object_list, obj, group=None):
        cmd = 'ALL_gather'
        tracker.trace_comm(C10Function, cmd, {'in': object_list, 'out': obj, 'group': group})
        if GLOBAL_CFG['c10_by_pass'] or torch_get_world_size(group=group) == GLOBAL_CFG.get('dummy_data_parallel_size', -1):
            if GLOBAL_CFG['c10_vit_real_tensor']:
                pass
            return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
        return torch_all_gather_object(object_list, obj, group=group)

    def local_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
        cmd = 'ALL_To_All'
        tracker.trace_comm(C10Function, cmd, {'in': input_tensor_list, 'out': output_tensor_list, 'group': group})
        if GLOBAL_CFG['c10_by_pass'] or torch_get_world_size(group=group) == GLOBAL_CFG.get('dummy_data_parallel_size', -1):
            if GLOBAL_CFG['c10_vit_real_tensor']:
                for i in range(len(output_tensor_list)):
                    if i < len(input_tensor_list):
                        output_tensor_list[i].copy_(input_tensor_list[i])
            if async_op:
                return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
            else:
                return
        return torch_all_to_all(output_tensor_list, input_tensor_list, group=group, async_op=async_op)

    def local_all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False):
        cmd = 'ALL_To_All'
        tracker.trace_comm(C10Function, cmd, {'in': input, 'out': output, 'group': group})
        if GLOBAL_CFG['c10_by_pass'] or torch_get_world_size(group=group) == GLOBAL_CFG.get('dummy_data_parallel_size', -1):
            if GLOBAL_CFG['c10_vit_real_tensor']:
                if input.shape[0] >= output.shape[0]:
                    output.copy_(input[:output.shape[0]])
                    if async_op:
                        return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
                    else:
                        return
                else:
                    print(input.shape[0], output.shape[0], input.shape[0] >= output.shape[0])
                rank = 0
                input_shape = input_split_sizes[rank]
                target_shape = [1 for i in output.shape]
                target_shape[0] = int(output.shape[0] / input_shape)
                output.copy_(torch.cat([input, input[:output.shape[0]-input.shape[0]]], dim=0))
                if async_op:
                    return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
                else:
                    return
                input_split_sizes = list(input_split_sizes)
                output_split_sizes = list(output_split_sizes)
                world_size = torch_get_world_size(group)
                if input_split_sizes is None:
                    split_size = input.shape[0] // world_size
                    input_split_sizes = [split_size] * world_size
                if output_split_sizes is None:
                    output_split_sizes = [input.shape[0] // world_size] * world_size

                input_chunks = list(torch.split(input, input_split_sizes, dim=0))  # [rank0_input, rank1_input, ...]

                recv_chunks = []
                for i in range(world_size):
                    chunk = input_chunks[i]  # 第 i 个 rank 的 input
                    if len(chunk.shape) == 1:
                        chunk_splits = torch.split(chunk, output_split_sizes)
                    else:
                        chunk_splits = torch.split(chunk, output_split_sizes, dim=0)
                    recv_chunks.append(chunk_splits[i])  # 当前 rank 从 i 拿第 i 段
                    output_data = torch.cat(recv_chunks, dim=0)
                    output.copy_(output_data)
                return
            else:
                return
        return torch_all_to_all_single(output, input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes, group=group, async_op=async_op)

    def local_broadcast(tensor, src, group=None, async_op=False):
        cmd = 'Broadcast'
        tracker.trace_comm(C10Function, cmd, {'out': tensor, 'group': group, 'src_rank': str(src)})
        if GLOBAL_CFG['c10_by_pass']:
            if async_op:
                return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
            else:
                return
        return torch_broadcast(tensor, src, group=group, async_op=async_op)

    def local_isend(tensor, dst, group=None, tag=0):
        cmd = 'isend'
        tracker.trace_comm(C10Function, cmd, {'out': tensor, 'dst': dst, 'group': group})
        if GLOBAL_CFG['c10_by_pass']:
            return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
        return torch_isend(tensor, dst, group, tag)

    def local_irecv(tensor, src=None, group=None, tag=0):
        cmd = 'irecv'
        tracker.trace_comm(C10Function, cmd, {'out': tensor, 'src': src, 'group': group})
        if GLOBAL_CFG['c10_by_pass']:
            return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
        return torch_irecv(tensor, src, group, tag)

    def local_batch_isend_irecv(op_list):
        cmd = 'batch_isend_irecv'
        tracker.trace_comm(C10Function, cmd, {'op_list': str(op_list)})
        if GLOBAL_CFG['c10_by_pass']:
            return [subprocess.Popen(["ncc_cmd={cmd}"], shell=True)]
        return torch_batch_isend_irecv(op_list)

    def local_P2POp(op, tensor, rank_list, group=None, tag=0):
        cmd = 'P2POp'
        tracker.trace_comm(C10Function, cmd, {'op': str(op), 'out': tensor, 'rank_list':  str(rank_list), 'group': group})
        return torch_P2POp(op, tensor, rank_list, group, tag)

    def local_barrier(*args, **kwargs):
        cmd = 'Barrier'
        tracker.trace_comm(C10Function, cmd, {'group': kwargs.get('group')})
        if GLOBAL_CFG.get('c10_by_pass'):
            return subprocess.Popen(["ncc_cmd={cmd}"], shell=True)
        return torch_barrier(*args, **kwargs)

    def local_get_process_group_ranks(*args, **kwargs):
        if GLOBAL_CFG.get('c10_by_pass'):
            if GLOBAL_CFG.get('virt_world_size', -1) != -1 or 'group' in kwargs and isinstance(kwargs['group'], list):
                return GLOBAL_CFG.get('virt_rank', 0)
        return torch_get_process_group_ranks(*args, **kwargs)

    def local_get_global_rank(*args, **kwargs):
        if GLOBAL_CFG.get('c10_by_pass'):
            if GLOBAL_CFG.get('virt_world_size', -1) != -1 or 'group' in kwargs and isinstance(kwargs['group'], list):
                return GLOBAL_CFG.get('virt_rank', 0)
        return torch_get_global_rank(*args, **kwargs)

    def local_get_world_size(*args, **kwargs):
        if GLOBAL_CFG.get('virt_world_size', -1) != -1:
            if 'group' in kwargs or len(args) > 0:
                if len(args) > 0:
                    return args[0].size()
                else:
                    return kwargs["group"].size()
                #return torch_get_world_size(*args, **kwargs)
            return GLOBAL_CFG.get('virt_world_size')
        if GLOBAL_CFG.get('c10_by_pass') and 'group' in kwargs and isinstance(kwargs['group'], list):
            return len(kwargs['group'])
        return torch_get_world_size(*args, **kwargs)

    def local_get_rank(*args, **kwargs):
        if GLOBAL_CFG.get('c10_by_pass'):
            return GLOBAL_CFG.get('virt_rank', 0)
        return torch_get_rank(*args, **kwargs)

    def local_is_initialized(*args, **kwargs):
        if GLOBAL_CFG.get('c10_by_pass') and not GLOBAL_CFG.get('dummy_data_parallel_size'):
            return True
        return torch_is_initialized(*args, **kwargs)

    def local_new_group(ranks=None, *args, **kwargs):
        rank = GLOBAL_CFG.get('virt_rank', -1)
        if rank != -1:
            world_size = len(ranks) if ranks else GLOBAL_CFG.get('virt_world_size')
            return new_group(rank, world_size, pg_options=kwargs.get('pg_options', None))
        return torch_new_group(*args, **kwargs)

    torch.distributed.all_reduce = partial(local_all_reduce)
    torch.distributed._reduce_scatter_base = partial(local_reduce_scatter_base)
    torch.distributed.reduce_scatter = partial(local_reduce_scatter)
    torch.distributed.reduce_scatter_tensor = partial(local_reduce_scatter_tensor)
    torch.distributed._all_gather_base = partial(local_all_gather_base)
    torch.distributed.all_gather = partial(local_all_gather)
    torch.distributed.all_gather_into_tensor = partial(local_all_gather_into_tensor)
    torch.distributed.all_gather_object = partial(local_all_gather_object)
    torch.distributed.all_to_all = partial(local_all_to_all)
    torch.distributed.all_to_all_single = partial(local_all_to_all_single)
    torch.distributed.broadcast = partial(local_broadcast)
    torch.distributed.isend = partial(local_isend)
    torch.distributed.irecv = partial(local_irecv)
    torch.distributed.batch_isend_irecv = partial(local_batch_isend_irecv)
    torch.distributed.P2POp = partial(local_P2POp)
    torch.distributed.get_world_size = partial(local_get_world_size)
    torch.distributed.get_rank = partial(local_get_rank)
    torch.distributed.is_initialized = partial(local_is_initialized)
    torch.distributed.get_process_group_ranks = partial(local_get_process_group_ranks)
    torch.distributed.get_global_rank = partial(local_get_global_rank)
    torch.distributed.barrier = partial(local_barrier)
    torch.distributed.new_group = partial(local_new_group)

    # 修复 torch._coalescing_manager 的 hang bug
    if hasattr(torch.distributed, '_coalescing_manager'):
        original_coalescing_manager = torch.distributed._coalescing_manager

        @contextlib.contextmanager
        def fixed_coalescing_manager(*args, **kwargs):
            if GLOBAL_CFG.get('c10_by_pass', False):
                # 使用简单的 mock manager 避免 hang
                class MockCoalescingManager:
                    def __init__(self):
                        self.works = []
                    def append(self, work):
                        if work:
                            self.works.append(work)
                    def wait(self):
                        # 安全地等待所有work完成，避免hang
                        for work in self.works:
                            if hasattr(work, 'wait'):
                                try:
                                    work.wait()
                                except:
                                    pass  # 忽略异常避免hang
                yield MockCoalescingManager()
            else:
                # 使用原始的 coalescing manager
                with original_coalescing_manager(*args, **kwargs) as cm:
                    yield cm

        torch.distributed._coalescing_manager = fixed_coalescing_manager


def new_group(group_rank, group_size, backend='gloo', store=None, pg_options=None):
    group_name = torch.distributed.distributed_c10d._process_group_name([k for k in range(group_size)], use_hashed_name=False)
    prefix_store = PrefixStore(f"{group_name}/", Store())
    if pg_options is None:
        try:
            pg_options = ProcessGroup.Options(backend=str(backend))
        except:
            pass
    if pg_options is None:
        return ProcessGroup(
            prefix_store, group_rank, group_size
        )
    else:
        return ProcessGroup(
            prefix_store, group_rank, group_size , pg_options
        )

def trace_comm(func=None):
    if GLOBAL_CFG.get('dummy_data_parallel_size'):
        GLOBAL_CFG['rank'] = int(os.getenv('RANK', '-1'))
        GLOBAL_CFG['world_size'] = int(os.getenv("WORLD_SIZE", '-1'))
        GLOBAL_CFG['group'] = new_group(GLOBAL_CFG['rank'], GLOBAL_CFG['dummy_data_parallel_size'])
    if not GLOBAL_CFG['comm_init']:
        register_comm_hook(func)
        GLOBAL_CFG['comm_init'] = True
    logging_set()

def trace_model(data, val):
    if GLOBAL_CFG.get('perf', False):
        return None
    all_modules(data, val, trace_module)
    if GLOBAL_CFG['nvtx']:
        return
    tracker.rank = torch.distributed.get_rank()
    for sc in torch.autograd.Function.__subclasses__():
        src_file = inspect.getfile(sc)
        if 'distributed' not in src_file:
            if any(k in sc.__name__ for k in ['BackwardHookFunction', 'record_function', '__new__', 'unpack_dual', \
                                              'apply', '__instancecheck__', '_tensor_or_tensors_to_tuple', '_make_grads', '_is_setup_context_defined', \
                                              '_engine_run_backward', 'grad', '_enter_inference_mode', 'MakeViewlessTensor']):
                continue
            print_rank_0('hook -->', 'api', '<==>', sc.__name__)
            func_forward = copy.deepcopy(sc.forward)
            func_backward = copy.deepcopy(sc.backward)
            sc.forward = partial(function_hook_fn(sc.__name__, func_forward, src_file))
            sc.backward = partial(function_hook_fn(sc.__name__, func_backward, src_file))

def load_model(package_main):
    import importlib
    package_spec = importlib.util.find_spec(package_main)
    if hasattr(package_spec, 'origin') and package_spec.origin.endswith('so'):
        print_rank_0('load so:', package_main, package_spec)
        if package_spec:
            package_modules = importlib.util.module_from_spec(package_spec)
            sys.modules[package_main] = package_modules
            package_spec.loader.exec_module(package_modules)
            return package_modules
        else:
            return None
    else:
        print_rank_0('load module:', package_main, package_spec)
        try:
            module = importlib.import_module(package_main)
        except:
            module = None
        if not module:
            items = package_main.split('.')
            package_name, module_name = '.'.join(items[:-1]), items[-1]
            package = importlib.import_module(package_name)
            module = getattr(package, module_name)
        return module

class Proxy:
    def __init__(self, obj, package_name, module_name, package_main):
        self._obj = obj
        self.package_name = package_name
        self.module_name = module_name
        self.package_main = package_main

    def __getattr__(self, name):
        # 获取原始方法
        attr = getattr(self._obj, name)
        if callable(attr):
            # 如果是方法，包装它
            return customer_op_hook_fn(f'{self.package_name}::', self.module_name, attr, self.package_main)
        else:
            # 如果不是方法，直接返回属性
            return attr

def each_file(dir, endwith='.so', with_path=False, only_file=False):
    paths = os.walk(dir)
    for path, dir_lst, file_lst in paths:
        for file_name in file_lst:
            if file_name.endswith(endwith):
                if with_path:
                    yield path, os.path.join(path, file_name)
                elif only_file:
                    yield file_name
                else:
                    yield os.path.join(path, file_name)

def load_and_trace(package_name, package_main):
    package_modules = load_model(package_main)
    if not package_modules:
        return
    if package_name != 'ops.aten':
        module_list = package_modules.__dict__.items()
    else:
        module_list = [(k, getattr(package_modules, k)) for k in dir(package_modules)]
    for module_name, func in module_list:
        if hasattr(func, '_op'):
            func = func._op
        if not module_name.startswith("__"):
            if any(k in str(func) for k in ['built-in method', 'function']):
                func_source = copy.deepcopy(func)
                try:
                    setattr(sys.modules[package_main], module_name,
                            partial(customer_op_hook_fn(f'{package_name}::', module_name, func_source, package_main)))
                    getattr(sys.modules[package_main], module_name).__name__ = module_name
                except:
                    getattr(package_modules, module_name)._op = partial(customer_op_hook_fn(f'{package_name}::',
                            module_name, func_source, package_main))
                print_rank_0('hook function:', package_modules, module_name)
            if package_name == 'deep_ep' and module_name == 'Buffer':
                func_source = copy.deepcopy(func)
                setattr(sys.modules[package_main], module_name,
                        Proxy(obj, package_name, module_name, package_main))
                proxied_obj = Proxy(func_source)
                print_rank_0('hook class:', package_modules, module_name)

def trace_triton_op(op, package_name=None):
    if op.__class__.__name__ in ['JITFunction']:
        op_name = op._fn_name if hasattr(op, '_fn_name') else op.fn.__name__
        print_rank_0('hook --> triton::', op_name, '<==>', op)
        op.add_pre_run_hook(triton_op_hook_fn(f'triton::', op_name))
        return op
    elif op.__class__.__name__ in ['function']:
        op_source = copy.deepcopy(op)
        if package_name:
            module = import_module(package_name)
            setattr(module, op.__name__, customer_op_hook_fn(f'triton::', op.__name__, op_source, 'triton'))
            return getattr(module, op.__name__)
    return None

def trace_customer_op(package_list):
    if GLOBAL_CFG.get('perf', False) or GLOBAL_CFG['nvtx']:
        return
    print_rank_0('hook package_list:', package_list)
    for package_name, package_main in package_list.items():
        te_path = ''
        if 'transformer_engine' in package_main:
            print_rank_0('_load_transformer_engine_torch start')
            try:
                te_path = _load_transformer_engine_torch()    # need to delete _load_library() in transformer_engine/pytorch/__init__.py
            except subprocess.CalledProcessError as e:
                print_rank_0('Warning: hook TE failed')
                package_main = ''
            try:
                import transformer_engine
                tracker.fw_version['te'] = 'te.' + transformer_engine.__version__.split('+')[0].replace('.', '_')
            except:
                pass
        if package_name == 'vllm':
            for module, op_name, func in _load_vllm_triton():
                print_rank_0('vllm:', op_name, func.__class__.__name__)
                try:
                    if func.__class__.__name__ in ['JITFunction']:
                        func.add_pre_run_hook(triton_op_hook_fn(f'vllm::', op_name))
                    elif func.__class__.__name__ in ['function']:
                        func_source = copy.deepcopy(func)
                        setattr(module, op_name, customer_op_hook_fn(f'vllm::', op_name, func_source, 'vllm'))
                except:
                    print_rank_0(f'Warning: hook vllm:{op_name} failed')
            continue
        if package_name == 'apex':
            print_rank_0('_load_apex_torch start')
            try:
                package_main = _load_apex_torch()
            except subprocess.CalledProcessError as e:
                print_rank_0('Warning: hook apex failed')
                package_main = ''
        if package_name == 'megatron':
            try:
                import megatron.core
                def local_jit_fuser(func):
                    print_rank_0('hook --> megatron::', func.__name__)
                    func = torch.compile(func)
                    return trace_triton_op(func, 'megatron.core.fusions')
                megatron.core.jit.jit_fuser = partial(local_jit_fuser)
            except:
                print_rank_0(f'Warning: hook megatron failed')
        if isinstance(package_main, list):
            for package_so in package_main:
                print_rank_0('found package:', package_so)
                try:
                    load_and_trace(f'{package_name}:{package_so}', package_so)
                except:
                    print_rank_0(f'Warning: hook {package_so} failed')
        elif package_main.startswith('path'):
            for file in each_file(package_main.split(':')[1], only_file=True):
                package_so = file.split('.')[0]
                print_rank_0('found package:', file, package_so)
                try:
                    load_and_trace(f'{package_name}:{package_so}', package_so)
                except:
                    print_rank_0(f'Warning: hook {package_so} failed')
        elif package_main:
            print_rank_0('found package:', package_main)
            try:
                load_and_trace(package_name, package_main)
            except:
                print_rank_0(f'Warning: hook {package_main} failed')
        if 'transformer_engine' in package_main and te_path:
            try:
                for module, op_name, func in _load_jit_library(te_path, ['jit_fuser', 'dropout_fuser']):
                    print_rank_0('triton:', op_name, func.__class__.__name__)
                    try:
                        if func.__class__.__name__ in ['JITFunction']:
                            func.add_pre_run_hook(triton_op_hook_fn(f'triton::', op_name))
                        elif func.__class__.__name__ in ['function']:
                            func_source = copy.deepcopy(func)
                            setattr(module, op_name, customer_op_hook_fn(f'triton::', op_name, func_source, 'transformer_engine'))
                    except:
                        print_rank_0(f'Warning: hook TE:{op_name} failed')
            except subprocess.CalledProcessError as e:
                print_rank_0('Warning: hook TE failed')

    GLOBAL_CFG['customer_op'] = True
