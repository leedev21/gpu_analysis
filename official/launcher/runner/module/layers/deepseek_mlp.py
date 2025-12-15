import torch
from torch import nn
import os, json
from typing import Optional, Union
from .quant import native_per_token_group_quant_fp8
from .native import RMSNorm
try:
    import vllm._custom_ops as ops
    from vllm.config import VllmConfig, ParallelConfig, CompilationConfig, set_current_vllm_config, get_current_vllm_config
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
except ImportError:
    print('vllm not found')
    ops = None

from .fusion import FusedDeepseekV2MoE
from vllm.forward_context import set_forward_context, get_forward_context

from .base import RandomDataset, get_dtype, generate_tensor_with_stats, pt_loader
from copy import deepcopy
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig


LOAD = False
LocalVllmBackend = {'backend': None, 'compiled': False, 'n': 0}


class LocalDataset(RandomDataset):
    def new(self, shape_list, dtype):
        if isinstance(shape_list[0], dict):
            for case in shape_list:
                if 'load' in case:
                    if len(case['load']) > 0 and isinstance(case['load'][0], tuple):
                        for pt_load in case['load']:
                            data = torch.load(pt_load[0][1], weights_only=False, map_location='cpu')
                            # print('in:', data)
                            self.data.append(data['inputs'][0])
                            # data = torch.load(pt_load[1][1], weights_only=False, map_location='cpu')
                            # print('out:', data)
                    else:
                        print('Error:', case['load'])
        else:
            for batch_size, seq_length, hidden_size in shape_list:
                if self.distribution:
                    hidden_states = generate_tensor_with_stats([batch_size*seq_length, hidden_size], dtype=dtype, device='cpu', args=self.distribution, seed=1)
                else:
                    scale = 1 / (hidden_size)
                    hidden_states = torch.randn(batch_size*seq_length, hidden_size, dtype=dtype) * scale

                self.data.append((hidden_states, ))


class FusedMoe(nn.Module):
    def __init__(self, _type, args):
        super(FusedMoe, self).__init__()
        global LocalVllmBackend
        self.__name__ = 'FusedMoe'
        self.args = args
        self.routed_scaling_factor = None
        self.log2phy = None
        self.prior_expert_map = None
        self.quant_config = None
        torch.set_default_dtype(get_dtype(precision=args['precision']))

        self.mlp = self.set_type(_type, args)
        self.post_attention_layernorm = None
        # exit()


    def set_type(self, _type, args):
        self.state = {}
        self._type = _type
        self.configs = get_config(_type, args)
        config = DictConfig(args['shape'])
        layer_idx  = config.layer_idx
        prior_expert_map = self.prior_expert_map
        log2phy =  self.log2phy
        self.routed_scaling_factor = config.routed_scaling_factor

        if 'has_rms_norm' in self.configs:
            self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)
            self.state['desp'] = f"Norm"
            self.state['op'] = f"Norm"
        else:
            self.state['desp'] = ""
            self.state['op'] = ""
        self.quant_config = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])

        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            layer_expert_map = (
                None if prior_expert_map is None else prior_expert_map[self.layer_idx]
            )
            layer_log2phy = (
                log2phy[self.layer_idx - config.first_k_dense_replace]
                if log2phy is not None
                else None
            )
            if 'fusion' in self.configs:
                self.state['op'] += f"DeepseekV2FusedMoE"
                self.state['desp'] += f"DeepseekV2FusedMoE"
                return FusedDeepseekV2MoE(
                    config=config,
                    quant_config=self.quant_config,
                    prefix=f"mlp",
                    layer_prior_expert_map=layer_expert_map,
                    layer_log2phy=layer_log2phy,
                )
            elif self.prior_expert_map or self.log2phy:
                self.state['op'] += f"DeepseekV2FusedMoE"
                self.state['desp'] += f"DeepseekV2FusedMoE"
                return DeepseekV2MoE(
                    config=config,
                    quant_config=self.quant_config,
                    prefix=f"mlp",
                    layer_prior_expert_map=layer_expert_map,
                    layer_log2phy=layer_log2phy,
                )
            else:
                self.state['op'] += f"DeepseekV2MoECuda"
                self.state['desp'] += f"DeepseekV2MoECuda"
                return DeepseekV2MoE(
                    config=config,
                    quant_config=self.quant_config,
                    prefix=f"mlp",
                    # enable_eplb=config.enable_eplb,
                )
        else:
            self.state['op'] += f"DeepseekV2MLP"
            self.state['desp'] += f"DeepseekV2MLP"
            return DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=self.quant_config,
                prefix=f"mlp",
            )

    @staticmethod
    def get_title(state):
        name = f"{state['op']}"
        for k in ['shape', 'dtype', 'hw_name']:
            if k in state:
                name += f"_{state[k]}"
        return name + f"_{LocalVllmBackend['n']}"

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        global LocalVllmBackend
        hidden_states = hidden_states.clone()
        self.state['device'] = hidden_states.device.type
        self.state['hw_name'] = self.state['device']
        self.state['shape'] = self.state['shape'] = 'x'.join([str(dim) for dim in hidden_states.shape])

        if 'has_rms_norm' in self.configs:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)
        if 'compile' in self.configs:
            with set_forward_context(None, get_current_vllm_config()):
                torch._dynamo.mark_dynamic(hidden_states, 0)
                if LocalVllmBackend['backend'] and not LocalVllmBackend['compiled']:
                    print('compile!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    if LocalVllmBackend['backend'] == 'graph':
                        self.mlp = torch.compile(
                                self.mlp, backend=LocalVllmBackend['backend'], options=options)   # "graph"
                    else:
                        self.mlp = torch.compile(
                                self.mlp, backend=LocalVllmBackend['backend'])
                    LocalVllmBackend['compiled'] = True
                hidden_states = self.mlp(hidden_states)
        else:
            with set_forward_context(None, get_current_vllm_config()):
                hidden_states = self.mlp(hidden_states)

        if isinstance(self.mlp,
                    DeepseekV2MLP) and hidden_states.dtype == torch.float16 \
                    and isinstance(hidden_states, torch.Tensor):
            # Fix FP16 overflow
            # Scaling the DeepseekV2MLP output, it is the input of
            # input_layernorm of next decoder layer.
            # The scaling of DeepseekV2MOE output would be done in the forward
            # of DeepseekV2MOE
            hidden_states *= 1. / self.routed_scaling_factor
        if 'save_output_by_pt' in self.args and self.args['save_output_by_pt']:
            torch.save({'name': self.__name__, 'state': self.state, 'd': hidden_states,
                        'desp': self.state.get('desp')},
                    f'data/pt/vllm/{self.get_title(self.state)}.pt')
        LocalVllmBackend['n'] += 1
        return hidden_states


def env(layer_type):
    pass


def get_config(layer_type, args):
    configs = args['config']
    return configs


def splite_test_case(args):
    case_dict = {'MOE': 'DeepseekV2MoE', 'MLP': 'DeepseekV2MLP'}
    if args['load_case_file']:
        pt_loader.load(args['load_case_file'], case_dict.values())
    main_type = None
    for types in args.module.layer.type:
        items = types.split('_')
        print(types, items)
        configs = []
        for typ in items:
            assert typ in args.module.layer.config
            if typ in case_dict:
                main_type = typ
            for config in args.module.layer.config[typ]:
                configs.append(config)
        if args['load_case_file']:
            total = 2
            for i, init in enumerate(args.module.layer.init):
                for _init, _input in pt_loader.get(case_dict[main_type]):
                    total -= 1
                    if total < 0:
                        return
                    yield {'type': types,
                        'input': {'load': _input},
                        'init': {'shape': init,
                                'config': configs,
                                'load': _init},
                        'use_vllm_backend': True
                        }
        else:
            for seq_length in args.module.layer.seq_length:
                for i, init in enumerate(args.module.layer.init):
                    print(types, int(seq_length), args.module.layer.input[i], list(init), configs)
                    yield {'type': types,
                        'seq_length': int(seq_length),
                        'input': args.module.layer.input[i],
                        'init': {'shape': init,
                                'config': configs},
                        'use_vllm_backend': True
                        }


module = None
def get_module(module_type, args):
    global module
    if not module:
        if args['load_case_file']:
            if 'load' in args:
                args['shape']['layer_idx'] = 0 if module_type == 'MLP' else 5
            module = FusedMoe(module_type, args)
            print('model:', module.mlp.state_dict().keys())
            if 'load' in args:
                data = torch.load(args['load'][0], weights_only=False)
                print('load:', data['init']['model']['_parameters'].keys())
                module.mlp.load_state_dict(data['init']['model']['_parameters'], strict=False)
                print('load success:', data['init']['name'])
            else:
                data = torch.load(os.path.join(args['load_case_file'], 'mlp_weight.pt'), weights_only=False)
                module.mlp.load_state_dict(data)
        elif LOAD:
            module = torch.load('data/pt/launcher_mlp_module.pt', weights_only=False)
            module.set_type(module_type, args)
        else:
            module = FusedMoe(module_type, args)
            # torch.save(module, 'data/pt/launcher_mlp_module.pt')
    else:
        module.set_type(module_type, args)
    return module


dataset = None
def data_loader(shape_list, n_samples, dtype, args):
    global dataset
    if not dataset:
        if args['load_case_file']:
            dataset = LocalDataset(shape_list, n_samples, dtype, args.task.acc._input_normal)
        elif LOAD:
            dataset = torch.load('data/pt/launcher_mlp_dataset.pt', weights_only=False)
        else:
            dataset = LocalDataset(shape_list, n_samples, dtype, args.task.acc._input_normal)
            # torch.save(dataset, 'data/pt/launcher_mlp_dataset.pt')
    return dataset