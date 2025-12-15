import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union
from .quant import native_per_token_group_quant_fp8
from .native import torch_moe, fused_moe_iterative
try:
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.fused_moe import fused_moe
except ImportError:
    print('vllm not found')
    vllm = None


try:
    import torch_custom_op_native
except ImportError:
    print('custom_op_native not found')
    torch_custom_op_native = None

from .base import RandomDataset, get_dtype
from copy import deepcopy


LOAD = True
SAVE_OUT = True


class LocalDataset(RandomDataset):
    def new(self, shape_list, dtype):
        for m, k, n, e, topk, ep_size, padding in shape_list:
            a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
            w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
            w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
            score = torch.randn((m, e), device="cuda", dtype=dtype)
            if ep_size > 1:
                local_e = e // ep_size
                e_ids = torch.randint(0,
                                    e, (local_e, ),
                                    device="cuda",
                                    dtype=torch.int32)
                e_map = torch.full((e, ), -1, device="cuda", dtype=torch.int32)
                e_map[e_ids] = torch.arange(local_e, device="cuda", dtype=torch.int32)
                w1 = w1[e_ids]
                w2 = w2[e_ids]
            else:
                e_map = None
            self.data.append((a, w1, w2, score, topk, e, e_map, padding))


class FusedMoe(nn.Module):
    def __init__(self, _type, args):
        super(FusedMoe, self).__init__()
        self.__name__ = 'FusedMoe'
        self.args = args
        self.set_type(_type)

    def set_type(self, _type):
        self.state = {}
        self._type = _type
        if _type == 'Vllm':
            self.apply = self.vllm_forward
        elif _type == 'Torch':
            self.apply = self.native_forward
        elif _type == 'Ref':
            self.apply = self.ref_forward
        else:
            print('No this kind of FusedMoe:', _type)
            exit(0)

    def get_title(self):
        name = f"{self.state['op']}"
        for k in ['device']:
            if k in self.state:
                name += f"_{self.state[k]}"
        return name

    def native_forward(self, a: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
             score: torch.Tensor, topk: int, e: int,
             e_map: Optional[torch.Tensor] = None, padding=False) -> torch.Tensor:
        self.state['op'] = 'native_torch_moe'
        out = torch_moe(a, w1, w2, score, topk, e_map)
        return out

    def ref_forward(self, a: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
             score: torch.Tensor, topk: int, e: int,
             e_map: Optional[torch.Tensor] = None, padding=False) -> torch.Tensor:
        self.state['op'] = 'fused_moe_iterative'
        out = fused_moe_iterative(a,
                                  w1,
                                  w2,
                                  score,
                                  topk,
                                  global_num_experts=e,
                                  expert_map=e_map,
                                  renormalize=False)
        return out

    def vllm_forward(self, a: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
             score: torch.Tensor, topk: int, e: int,
             e_map: Optional[torch.Tensor] = None, padding=False) -> torch.Tensor:
        if padding:
            w1 = F.pad(w1, (0, 128), "constant", 0)[..., 0:-128]
            torch.cuda.empty_cache()
            w2 = F.pad(w2, (0, 128), "constant", 0)[..., 0:-128]
            torch.cuda.empty_cache()
            self.state['op'] = 'fused_moe_padding'
        else:
            self.state['op'] = 'fused_moe'

        out = fused_moe(a,
                        w1,
                        w2,
                        score,
                        topk,
                        global_num_experts=e,
                        expert_map=e_map,
                        renormalize=False)
        return out

    def forward(self, a, w1, w2, score, topk, e, e_map, padding):
        self.state['device'] = a.device.type
        out = self.apply(a, w1, w2, score, topk, e, e_map, padding)
        if not self.args['forward_only']:
            loss = out.sum()
            return loss
        if SAVE_OUT:
            torch.save({'name': self.__name__, 'state': self.state, 'd': out}, f'data/pt/vllm/{self.get_title()}.pt')
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


module = None
def get_module(module_type, args):
    global module
    if not module:
        if LOAD:
            module = torch.load('data/pt/launcher_fused_moe_module.pt', weights_only=False)
            module.set_type(module_type)
        else:
            module = FusedMoe(module_type, args)
            torch.save(module, 'data/pt/launcher_fused_moe_module.pt')
    else:
        module.set_type(module_type)
    return module


dataset = None
def data_loader(shape_list, n_samples, dtype):
    global dataset
    if not dataset:
        if LOAD:
            dataset = torch.load('data/pt/launcher_fused_moe_dataset.pt', weights_only=False)
        else:
            dataset = LocalDataset(shape_list, n_samples, dtype)
            torch.save(dataset, 'data/pt/launcher_fused_moe_dataset.pt')
    return dataset