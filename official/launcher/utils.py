import os
import sys
import torch
import argparse
from importlib import import_module
from collections import namedtuple, OrderedDict
import time


DATA_PT_PATH = 'data/pt'
if not os.path.isdir(DATA_PT_PATH):
    os.makedirs(DATA_PT_PATH, exist_ok=True)


def get_device():
    return 'cuda'

def env(layer_type):
    config = {
        'TE unfused': {'NVTE_FLASH_ATTN': 0, 'NVTE_FUSED_ATTN': 0, 'NVTE_MASKED_SOFTMAX_FUSION': 0},
        'TE fused': {'NVTE_FLASH_ATTN': 0, 'NVTE_FUSED_ATTN': 1, 'NVTE_MASKED_SOFTMAX_FUSION': 1},
        'TE flash-attn': {'NVTE_FLASH_ATTN': 1, 'NVTE_FUSED_ATTN': 0, 'NVTE_MASKED_SOFTMAX_FUSION': 0},
        'TE fuse softmax': {'NVTE_FLASH_ATTN': 0, 'NVTE_FUSED_ATTN': 0, 'NVTE_MASKED_SOFTMAX_FUSION': 1},
        'Megatron': {},
        'Torch': {},
    }
    os.environ['API_USER'] = 'False'
    for k, v in config[layer_type].items():
        os.environ[k] = str(v)
        print('Env Set:', k, v)


def Event():
    if torch.cuda.is_available():
        return torch.cuda.Event(enable_timing=True)
    else:
        return None


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def record(timer_start, timer_end=None):
    if torch.cuda.is_available():
        if not timer_end:
            timer_start.record(stream=torch.cuda.current_stream())
        else:
            timer_end.record(stream=torch.cuda.current_stream())
            timer_end.synchronize()
            step_time_cuda = timer_start.elapsed_time(timer_end)
            return step_time_cuda
    return 0.0


def print_rank_0(*args):
    if torch.distributed.is_initialized():
        rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
    if rank == 0:
        print(*args)


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


def _get_env_cfg():
    n_device = torch.cuda.device_count()
    print('available devices:', n_device, torch.cuda.is_available())
    hw_0 = torch.cuda.get_device_name(0)
    print(f"Device 0: {hw_0}")
    for i in range(n_device):
        hw_tmp = torch.cuda.get_device_name(i)
        if (hw_0 != hw_tmp):
            print('warning device:{} and device:0 have different names!'.format(i), hw_tmp)
    if "Device" == hw_0:
        prop = torch.device.get_device_properties(0)
        hw_0 = str(prop.major)
    hw_name_dict = {
        # GPU name
        "NVIDIA B200": "B200",
        "NVIDIA B100": "B100",
        "NVIDIA H200SXM": "H200SXM",
        "NVIDIA H100 Tensor Core GPU SXM": "H100SXM",
        "NVIDIA H100 PCIe": "H100PCIE",
        "NVIDIA A100-SXM4-40GB": "A100SXM",
        "NVIDIA A100 Tensor Core GPU PCIe": "A100PCIE",
        "NVIDIA A100-PCIE-40GB": "A100PCIE",
        "NVIDIA L40S": "L40S",
        "NVIDIA H20": "H20SXM",
        "NVIDIA GeForce RTX 3090": "3090",
        "NVIDIA GeForce RTX 5090": "5090",
    }
    hw = hw_name_dict.get(hw_0, "None")
    if hw == "None":
        print(hw_0, 'not found, please add it to hw_name_dict')
    return n_device, hw
