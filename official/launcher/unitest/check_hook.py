import os
import sys
from importlib import import_module
from functools import partial
from torchtrace.utils import read_file
from torchtrace.hooks import GLOBAL_CFG, _load_transformer_engine_torch, each_file, load_and_trace, load_model, customer_op_hook_fn

def trace_customer_op(package_list):
    if GLOBAL_CFG.get('perf', False) or GLOBAL_CFG['nvtx']:
        return
    print('hook package_list:', package_list)
    for package_name, package_main in package_list.items():
        if 'transformer_engine' in package_main:
            print('_load_transformer_engine_torch start')
            try:
                _load_transformer_engine_torch()    # need to delete _load_library() in transformer_engine/pytorch/__init__.py
            except:
                print('Warning: hook TE failed')
        if package_main.startswith('path'):
            for file in each_file(package_main.split(':')[1], only_file=True):
                package_so = file.split('.')[0]
                print('found so:', file, package_so)
                load_and_trace(f'{package_name}:{package_so}', package_so)
        else:
            load_and_trace(package_name, package_main)
    package_modules = load_model('vllm._C')
    print(package_modules)
    for module_name, func in package_modules.__dict__.items():
        print(module_name, func)
        if not module_name.startswith("__"):
            if 'built-in method' in str(func):
                print(func)
    # load_and_trace('vllm', 'vllm._C')
    import torch
    import vllm._C
    import vllm._moe_C
    import vllm._custom_ops
    import vllm.attention.ops.prefix_prefill
    import copy
    lines = read_file(os.path.dirname(__file__), 'vllm_ops.txt')
    for line in lines:
        print(line.strip())
        # module = import_module('torch.ops._C')
        if not str(func).endswith('default'):
            try:
                func = getattr(torch.ops._C, line.split('.')[-1])
            except:
                func = None
        else:
            func = None
        if func:
            func_source = copy.deepcopy(func)
            customer_op_hook_fn(f'{package_name}::', module_name, func_source, package_main)
            print('hook setattr', package_modules, module_name)
            print('True')
        else:
            print('False')
    GLOBAL_CFG['customer_op'] = True


trace_customer_op({# 'apex': 'path:/opt/conda/lib/python3.10/site-packages',
                   'te': 'transformer_engine_torch',
                   'flash_attn': 'flash_attn_2_cuda'})