####################################################
# Use Guide:
# from dryrun_backend import MyTestMode
# with MyTestMode():
#    # your code here
# ###################################################

from torchtrace.torch_api import api_trace_start, api_trace_end
from torchtrace.torch_dispatch import dispatch_trace_start, dispatch_trace_end
from torchtrace.hooks import (trace_model, trace_optm, remove_hook, trace_comm,
                              trace_customer_op, trace_triton_op)
from torchtrace.module import tracker
from torchtrace.utils import GLOBAL_CFG, DEFAULT_DATA_PATH, load_default
import os
import yaml


def set_torchtrace(torch_dispatch_trace: bool = True,
                   torch_api_trace: bool = True,
                   save_pt: bool = False,
                   sync_mode: bool = False,
                   save_to: str = ''):
    global GLOBAL_CFG
    GLOBAL_CFG['sync_mode'] = sync_mode
    GLOBAL_CFG['save_pt'] = save_pt
    if GLOBAL_CFG.get('perf') or GLOBAL_CFG.get('nvtx'):
        return

    if torch_dispatch_trace == True:
        dispatch_trace_start(save_pt)
    else:
        dispatch_trace_end()

    if torch_api_trace == True:
        api_trace_start()
    else:
        api_trace_end()
        # remove_hook()

    if save_to:
        GLOBAL_CFG['data_pt_path'] = save_to.rstrip('/')


def update(attr, val=None, data=None, **kwargs):
    print(f'[TorchTrace] Update called with attr: {attr}, val: {val}, data: {data}, kwargs: {kwargs}')
    if attr == 'state':
        if data and 'optim:step' in data:
            trace_optm(val, data)
        else:
            GLOBAL_CFG[attr] = val
            if data == 'step':
                GLOBAL_CFG['step'] += 1
    elif attr == 'filter':
        if data == 'default':
            GLOBAL_CFG[val] = load_default(f"{val}_default.yaml")
            for item in GLOBAL_CFG[val]['op']:
                print('op trace for pt save default:', item)
        else:
            # Parse custom configuration as YAML
            try:
                parsed_config = yaml.safe_load(data)
                if val == "op_trace":
                    trace_type = "op"
                elif val == "module_trace":
                    trace_type = "module"
                else:
                    raise Exception(f"Unknown trace type: {val}")
                # Load default config
                default_config = load_default(f"{trace_type}_trace_default.yaml")

                if parsed_config.get('op') == 'default' or parsed_config.get('module') == 'default':
                    final_config = default_config.copy()
                    for key in parsed_config:
                        if key != 'op' and key != 'module':
                            final_config[key] = parsed_config[key]
                else:
                    final_config = parsed_config.copy()
                    for key in default_config:
                        if key not in final_config:
                            final_config[key] = default_config[key]

                GLOBAL_CFG[val] = final_config
                print(f'[TorchTrace] Loaded custom config for {val}: {parsed_config}')
            except yaml.YAMLError as e:
                print(f'[TorchTrace] Error parsing config "{data}" as YAML: {e}')
                print(f'[TorchTrace] Falling back to default config')
                GLOBAL_CFG[val] = load_default(f"{val}_default.yaml")
    elif attr == 'model':
        trace_model(data, val)
    elif attr == 'step':
        tracker.set_step(step=data, state=val, **kwargs)
    elif attr == 'iter':
        tracker.set_iter(iter=data, state=val)
    elif attr == 'optim':
        trace_optm(val)
    elif attr == 'comm':
        if val == 'by_pass':
            GLOBAL_CFG['c10_by_pass'] = True
        else:
            GLOBAL_CFG[val] = data
        rank = int(os.getenv("CH_VIRT_RANK", '-1'))
        world_size = int(os.getenv("CH_VIRT_WORLD_SIZE", '-1'))
        if world_size != -1 and rank != -1:
            GLOBAL_CFG['virt_rank'] = rank
            GLOBAL_CFG['virt_world_size'] = world_size
        trace_comm(data)
    elif attr == 'customer_op':
        if val is not None:
            trace_customer_op(val)
        elif not GLOBAL_CFG.get('customer_op', False):
            update('customer_op', {'apex': '', 'nn.functional': 'torch.nn.functional',
                                   'ops.aten': 'torch.ops.aten', 'vllm': '',
                                   'deep_gemm': 'deep_gemm', 'te': 'transformer_engine_torch',
                                   'megatron': '', 'flash_attn': 'flash_attn_2_cuda'})
    elif attr == 'args':
        from omegaconf import OmegaConf
        import argparse
        import torch
        out_path = os.getenv('NSYS_OUT') if os.path.exists(os.getenv('NSYS_OUT')) else GLOBAL_CFG['data_pt_path']
        if os.path.exists(out_path):
            hydra_path = os.path.join(out_path, 'hydra.yaml')
            if isinstance(val, argparse.Namespace):
                args_dict = vars(val)
                common_cfg = {'base': {}}
                for k, v in args_dict.items():
                    if isinstance(v, torch.dtype):
                        v = str(v)
                    common_cfg['base'][k] = v
                OmegaConf.save(common_cfg, hydra_path)
    elif attr == 'triton_op':
        trace_triton_op(val)
    else:
        GLOBAL_CFG[attr] = val
