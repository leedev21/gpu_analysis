import torch
from torch import nn
import os, json
from typing import Optional, Union
from .quant import native_per_token_group_quant_fp8
from .native import baseline_scaled_mm
from .triton import _w8a8_block_fp8_matmul
try:
    import vllm._custom_ops as ops
except ImportError:
    print('vllm not found')
    ops = None
try:
    import torch_custom_op_native
except ImportError:
    print('custom_op_native not found')
    torch_custom_op_native = None

from .base import RandomDataset, get_dtype, generate_tensor_with_stats
from copy import deepcopy


LOAD = False
SAVE_OUT = True

class LocalDataset(RandomDataset):
    def new(self, shape_list, dtype):
        for num_tokens, hidden_size, quant_dtype, add_residual, scale_ub in shape_list:
            if self.distribution:
                x = generate_tensor_with_stats([num_tokens, hidden_size], dtype=dtype, device='cpu', args=self.distribution, seed=1)
                residual = generate_tensor_with_stats([num_tokens, hidden_size], dtype=dtype, device='cpu', args=self.distribution, seed=0)
            else:
                scale = 1 / (hidden_size)
                x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
                residual = torch.randn_like(x) * scale if add_residual else None
            quant_dtype = get_dtype(precision=quant_dtype)
            # rms_x, _ = self.ref_rms_norm(layer, x, residual)
            # scale_ub = torch.mean(rms_x).to(dtype=torch.float32, device='cuda')

            self.data.append((x, quant_dtype, residual, scale_ub))

    # @staticmethod
    # def ref_rms_norm(rms_norm_layer: RMSNorm,
    #                 x: torch.Tensor,
    #                 residual: Optional[torch.Tensor]) \
    #         -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    #     if residual is not None:
    #         residual = residual.clone()
    #         out, residual = rms_norm_layer.forward_native(x, residual)
    #     else:
    #         out = rms_norm_layer.forward_native(x)
    #     return out, residual


class CustomerLinear(nn.Module):
    def __init__(self, _type, args):
        super(RmsNormLinear, self).__init__()
        self.__name__ = 'RmsNormLinear'
        self.args = args
        self.configs = args['config']
        self.eps = args['shape'][1]
        self.group_size = 128
        self.use_native_quant_cpu = False
        dtype = get_dtype(precision=args['precision'])
        self.module = RMSNorm(
                    *args['shape']
            ).to(dtype)
        if args.get('distribution'):
            self.module.weight.data = generate_tensor_with_stats([args['shape'][0]],
                                                                 dtype=dtype,
                                                                 device='cpu',
                                                                 args=args['distribution'], seed=2)
        else:
            self.module.weight.data.normal_(mean=1.0, std=0.1)
        self.set_type(_type, args)

    def set_type(self, _type, args):
        self.state = {}
        self._type = _type
        self.configs = get_config(_type, args)
        # self._type = ['cpu', 'use_native_quant_cpu', 'use_quant_lib', 'use_native_layer_norm']
        if 'use_fused_op' in self.configs:
            self.apply = self.vllm_forward
        elif 'enable_quant' in self.configs:
            self.apply = self.ref_forward
        else:
            self.apply = self.rms_norm_forward

    @staticmethod
    def get_title(state):
        name = f"{state['op']}"
        for k in ['shape', 'dtype', 'quant', 'residual', 'scale_ub', 'hw_name']:
            if k in state:
                name += f"_{state[k]}"
        return name

    @staticmethod
    def native_forward(rms_norm_layer,
                 x: torch.Tensor,
                 residual: Optional[torch.Tensor],
                 dtype: Optional[torch.dtype] = None) \
        -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
        _w8a8_block_fp8_matmul(a, b, scale_a, scale_b, out_dtype, bias)

        return out, residual

    def rms_norm_forward(self,
                 x: torch.Tensor,
                 quant_dtype: torch.dtype,
                 residual: Optional[torch.Tensor],
                 scale_ub: Optional[torch.Tensor],
                 residual_add_dtype: Optional[torch.dtype] = None) \
        -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        rms_norm_layer = self.module

        # Norm
        self.state['op'] = 'rms_norm_reference'
        if self.state['device'] == 'cpu' or residual_add_dtype is not None:
            self.state['dtype'] = f"{x.dtype}".replace('torch.', '')
        if residual is not None:
            residual = residual.clone()
        weight = self.module.weight
        if 'use_torch_native' in self.configs:
            out, residual = self.native_forward(rms_norm_layer, x, residual, residual_add_dtype)
        elif 'use_custom_op_native' in self.configs:
            torch.ops._VLLM_C.fused_add_rms_norm(x, residual, weight, self.eps)
            out = x.clone()
        elif 'use_vllm' in self.configs:
            ops.fused_add_rms_norm(x, residual, weight, self.eps)
            out = x.clone()
        else:
            print('No this kind of linear:', self._type)

        return out, residual

    def ref_forward(self, x: torch.Tensor,
                                quant_dtype: torch.dtype,
                                residual: Optional[torch.Tensor],
                                scale_ub: Optional[torch.Tensor],
                                residual_add_dtype: Optional[torch.dtype] = None) \
        -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if scale_ub is not None:
            assert quant_dtype == torch.float8_e4m3fn

        # Norm
        torch_out, residual = self.rms_norm_forward(x, quant_dtype, residual, scale_ub, residual_add_dtype=residual_add_dtype)

        if not residual_add_dtype:
            residual_add_dtype = torch.float32

        # Quant
        if quant_dtype == torch.float8_e4m3fn:
            if 'use_torch_native_quant' in self.configs:
                self.state['quant'] = 'native_quant'
                self.state['desp'] = f"Realized by torch native RMSNorm + native_per_token_group_quant_fp8. \
device on {self.state['hw_name']}. \
torch native RMSNorm(from Vllm): input with {x.dtype}, residual add with {residual_add_dtype}, run with {x.dtype}. \
native_per_token_group_quant_fp8(from Vllm): compute with {torch.float32}. \
The function returns a tuple containing the following elements: \
(0) **Output**: type e4m3fn; \
(1) **Scales**: type fp32; \
(2) **Residual (Optional)**: type {x.dtype}."
                torch_out, scales = native_per_token_group_quant_fp8(torch_out, self.group_size)
            elif 'use_vllm' in self.configs:
                self.state['quant'] = 'vllm_quant'
                self.state['desp'] = f"Realized by torch native RMSNorm + vllm._custom_ops.scaled_fp8_quant. \
device on {self.state['hw_name']}. \
torch native RMSNorm: input with {x.dtype}, residual add with {residual_add_dtype}, run with {x.dtype}. \
vllm._custom_ops.scaled_fp8_quant: compute with {torch.float32}. \
The function returns a tuple containing the following elements: \
(0) **Output**: type e4m3fn; \
(1) **Scales**: type fp32; \
(2) **Residual (Optional)**: type {x.dtype}."
                torch_out, scales = ops.scaled_fp8_quant(torch_out.float(),
                                                        scale_ub=scale_ub,
                                                        use_per_token_if_dynamic=True)
            elif 'use_custom_op_native' in self.configs:
                scale_shape = list(x.shape)
                scale_shape[-1] = scale_shape[-1] // self.group_size
                out = torch.zeros_like(x, dtype=torch.bfloat16, device=x.device).to(quant_dtype)
                scale_out = torch.zeros(scale_shape, dtype=torch.float32, device=x.device)
                self.state['quant'] = 'op_lib_quant'
                self.state['desp'] = f"Realized by torch native RMSNorm + torch.ops._C.dynamic_per_token_group_fp8_quant. \
device on {self.state['hw_name']}. \
torch native RMSNorm: input with {x.dtype}, residual add with {residual_add_dtype}, run with {x.dtype}. \
torch.ops._C.dynamic_per_token_group_fp8_quant(Ourself): compute with {torch.float32}. \
The function returns a tuple containing the following elements: \
(0) **Output**: type e4m3fn; \
(1) **Scales**: type fp32; \
(2) **Residual (Optional)**: type {x.dtype}."
                torch.ops._VLLM_C.dynamic_per_token_group_fp8_quant(out,
                                                        scale_out,
                                                        torch_out.to(torch.bfloat16),
                                                        self.group_size)
                torch_out = out.clone()
                scales = scale_out.clone()
        else:
            assert quant_dtype == torch.int8
            self.state['quant'] = 'int8_quant'
            torch_out, scales = ops.scaled_int8_quant(torch_out)

        return torch_out.detach().cpu(), scales.detach().cpu(), residual.detach().cpu() if residual is not None else residual

    def vllm_forward(self, x: torch.Tensor,
                                quant_dtype: torch.dtype,
                                residual: Optional[torch.Tensor],
                                scale_ub: Optional[torch.Tensor],
                                residual_add_dtype: Optional[torch.dtype] = None) \
        -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        weight = self.module.weight
        if residual is not None:
            residual = residual.clone()
        self.state['dtype'] = f"{x.dtype}".replace('torch.', '')
        if x.device.type == 'cuda' and not 'use_custom_op_native' in self.configs:
            self.state['op'] = 'rms_norm_dynamic_per_token_quant'
            self.state['desp'] = f"Called to vllm._custom_ops.rms_norm_dynamic_per_token_quant \
device on {self.state['hw_name']}. \
vllm._custom_ops.rms_norm_dynamic_per_token_quant: input with {x.dtype}. \
The function returns a tuple containing the following elements: \
(0) **Output**: type e4m3fn; \
(1) **Scales**: type fp32; \
(2) **Residual (Optional)**: type {x.dtype}."
            out, scales = ops.rms_norm_dynamic_per_token_quant(x, weight, self.eps,
                                                            quant_dtype, scale_ub,
                                                            residual)
        else:
            out = torch.zeros_like(x, dtype=torch.bfloat16, device=x.device).to(quant_dtype)
            # residual = torch.zeros_like(x, dtype=torch.bfloat16, device=x.device)
            # self.state['residual'] = 'residual_zero'
            scale_shape = list(x.shape)
            scale_shape[-1] = scale_shape[-1] // self.group_size
            scales = torch.zeros(scale_shape, dtype=torch.float32, device=x.device)
            if residual is not None:
                self.state['op'] = 'fused_add_rms_norm_per_token_group_quant_fp8'
                self.state['desp'] = f"Called to torch.ops._C.fused_add_rms_norm_per_token_group_quant_fp8 \
device on {self.state['hw_name']}. \
torch.ops._C.fused_add_rms_norm_per_token_group_quant_fp8 : input with {x.dtype}. \
The function returns a tuple containing the following elements: \
(0) **Output**: type e4m3fn; \
(1) **Scales**: type fp32; \
(2) **Residual (Optional)**: type {x.dtype}."

                torch.ops._VLLM_C.fused_add_rms_norm_per_token_group_quant_fp8(out, residual, scales, x, weight, self.eps, self.group_size)
            else:
                self.state['op'] = 'rms_norm_per_token_group_quant_fp8'
                self.state['desp'] = f"Called to torch.ops._C.rms_norm_per_token_group_quant_fp8 \
device on {self.state['hw_name']}. \
torch.ops._C.rms_norm_per_token_group_quant_fp8 : input with {x.dtype}. \
The function returns a tuple containing the following elements: \
(0) **Output**: type e4m3fn; \
(1) **Scales**: type fp32; \
(2) **Residual (Optional)**: type {x.dtype}."
                torch.ops._VLLM_C.rms_norm_per_token_group_quant_fp8(out, scales, x, weight, self.eps, self.group_size)

        return out.detach().cpu(), scales.detach().cpu(), residual.detach().cpu()

    def forward(self, x, quant_dtype, residual, scale_ub):
        # if LOAD and self._type == 'Torch':
        #     return torch.load('data/pt/launcher_out.pt', weights_only=False)
        device = 'cuda'
        residual_add_dtype = None
        if 'CPU' in self._type:
            device = 'cpu'
            dtype = torch.float64 # torch.float32
            residual_add_dtype = torch.float64
            x = x.to(dtype=dtype, device=device)
            if residual is not None:
                residual = residual.to(dtype=dtype, device=device)
            self.module.to(device=device)
        if scale_ub is not None:
            rms_x, _ = self.native_forward(self.module, x, residual)
            if self._type == 'torch':
                scale_ub = torch.mean(rms_x).to(dtype=dtype, device=device)
            else:
                scale_ub = torch.mean(rms_x).to(dtype=torch.float32, device=device)
            # self.state['scale_ub'] = 'scale_ub'

        self.state['device'] = x.device.type
        self.state['hw_name'] = self.state['device']
        self.state['shape'] = self.state['shape'] = 'x'.join([str(dim) for dim in x.shape])
        out = self.apply(x, quant_dtype, residual, scale_ub, residual_add_dtype=residual_add_dtype)
        if not self.args['forward_only']:
            loss = out.sum()
            return loss
        if SAVE_OUT:
            torch.save({'name': self.__name__, 'state': self.state, 'd': out,
                        'desp': self.state.get('desp')},
                       f'data/pt/vllm/{self.get_title(self.state)}.pt')
        return out


def env(layer_type):
    pass


def get_config(layer_type, args):
    # items = layer_type.split('_')
    configs = args['config']
    # print(layer_type, args)
    # for typ in items:
    #     assert typ in args['config']
    #     for config in args['config'][typ]:
    #         if config == 'use_custom_op_native' and not torch_custom_op_native:
    #             continue
    #         if config == 'use_vllm' and not ops:
    #             continue
    #         configs.append(config)
    return configs


def splite_test_case(args):
    for types in args.module.layer.type:
        items = types.split('_')
        print(items)
        configs = []
        for typ in items:
            assert typ in args.module.layer.config
            print(args.module.layer.config[typ])
            for config in args.module.layer.config[typ]:
                if config == 'use_custom_op_native' and not torch_custom_op_native:
                    continue
                if config == 'use_vllm' and not ops:
                    continue
                configs.append(config)
        for seq_length in args.module.layer.seq_length:
            for i, init in enumerate(args.module.layer.init):
                print(types, int(seq_length), args.module.layer.input[i], list(init), configs)
                yield {'type': types,
                    'seq_length': int(seq_length),
                    'input': args.module.layer.input[i],
                    'init': {'shape':list(init),
                             'config': configs}
                    }


module = None
def get_module(module_type, args):
    global module
    if not module:
        if args['load_case_file']:
            module = RmsNormLinear(module_type, args)
            module.module.weight.data = torch.load(os.path.join(args['load_case_file'], 'rms_norm_weight.pt'), weights_only=False)['weight']
        elif LOAD:
            module = torch.load('data/pt/launcher_rms_norm_module.pt', weights_only=False)
            module.set_type(module_type, args)
        else:
            module = RmsNormLinear(module_type, args)
            torch.save(module, 'data/pt/launcher_rms_norm_module.pt')
    else:
        module.set_type(module_type, args)
    return module


dataset = None
def data_loader(shape_list, n_samples, dtype, args):
    global dataset
    if not dataset:
        if args['load_case_file']:
            dataset = LocalDataset(shape_list, n_samples, dtype, args.task.acc._input_normal)
            x, quant_dtype, residual, scale_ub = dataset.data[0]
            data = torch.load(os.path.join(args['load_case_file'], 'dataset_bs64.pt'), weights_only=False)
            dataset.data = [(x, quant_dtype, residual, scale_ub) for x, residual in data['input']]
        elif LOAD:
            dataset = torch.load('data/pt/launcher_rms_norm_dataset.pt', weights_only=False)
        else:
            dataset = LocalDataset(shape_list, n_samples, dtype, args.task.acc._input_normal)
            torch.save(dataset, 'data/pt/launcher_rms_norm_dataset.pt')
    return dataset