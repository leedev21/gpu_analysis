import torch
from torch import nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from .base import RandomDataset


def get_dtype(args):
    if args in [32, 'fp32']:
        return torch.float
    if args in [16, 'fp16']:
        return torch.half
    if args in ['bf16']:
        return torch.bfloat16
    return None


class GPTLinear(nn.Module):
    def __init__(self, linear_type, args):
        super(GPTLinear, self).__init__()
        self.__name__ = 'linear'
        self.args = args
        if linear_type == 'TE':
            model = te.Linear
        elif linear_type == 'Torch':
            model = nn.Linear
        else:
            print('No this kind of linear:', linear_type)
            exit(0)
        self.linear = model(
                    *args['shape']
            ).to(get_dtype(args['precision']))
        if args['precision'] == 'fp8':
            self.enable_fp8 = True
            self.fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)
        else:
            self.enable_fp8 = False

    def forward(self, inp):
        if self.enable_fp8:
            with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):
                out = self.linear(inp)
        else:
            out = self.linear(inp)
        if not self.args['forward_only']:
            loss = out.sum()
            return loss
        return out


def env(layer_type):
    pass


def splite_test_case(args):
    for typ in args.module.layer.type:
        for seq_length in args.module.layer.seq_length:
            for init in args.module.layer.init:
                yield {'type': typ,
                       'seq_length': int(seq_length),
                       'init': {'shape':list(init)}
                      }


def get_module(module_type, args):
    return GPTLinear(module_type, args)


data_loader = RandomDataset