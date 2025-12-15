import torch
from torch import nn
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from .base import RandomDataset


def get_dype(args):
    if args in [32, 'fp32']:
        return torch.float
    if args in [16, 'fp16']:
        return torch.half
    if args in ['bf16']:
        return torch.bfloat16
    return None


class Attn(nn.Module):
    def __init__(self, linear_type, args):
        super(Attn, self).__init__()
        self.args = args

    def forward(self, qkv):
        out = flash_attn_qkvpacked_func(qkv, 0.1, causal=True)
        return out


def env(layer_type):
    pass


def splite_test_case(args):
    for typ in args.module.layer.type:
        for seq_length in args.module.layer.seq_length:
            yield {'type': typ,
                    'seq_length': int(seq_length),
                    'init': {}
                    }


def get_module(module_type, args):
    return Attn(module_type, args)


data_loader = RandomDataset