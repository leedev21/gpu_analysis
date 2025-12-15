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
    from vllm.distributed.parallel_state import init_distributed_environment, initialize_model_parallel
    init_distributed_environment(world_size=1, rank=0, distributed_init_method="env://", local_rank=0, backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
except ImportError:
    print('vllm not found')
    ops = None

from vllm.model_executor.models.deepseek_v2 import DeepseekV2Attention
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention
DeepseekV2MLAAttentionFusion = None
from vllm.forward_context import set_forward_context, get_forward_context
from vllm.attention import AttentionMetadata, AttentionMetadataBuilder

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
                            self.data.append((data['inputs'][1]['positions'], data['inputs'][1]['hidden_states'], ))
                            # data = torch.load(pt_load[1][1], weights_only=False, map_location='cpu')
                            # print('out:', data)
                    else:
                        print('Error:', case['load'])
        else:
            for batch_size, num_tokens, hidden_size, max_position, add_residual in shape_list:
                print(batch_size, num_tokens, hidden_size, max_position, add_residual)
                positions = torch.randint(0, max_position, (1, num_tokens))
                if self.distribution:
                    hidden_states = generate_tensor_with_stats([num_tokens, hidden_size], dtype=dtype, device='cpu', args=self.distribution, seed=1)
                    residual = generate_tensor_with_stats([num_tokens, hidden_size], dtype=dtype, device='cpu', args=self.distribution, seed=0)
                else:
                    scale = 1 / (hidden_size)
                    hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
                    residual = torch.randn_like(x) * scale if add_residual else None

                self.data.append((positions, hidden_states, residual, ))


class Attention(nn.Module):
    def __init__(self, _type, args):
        super(Attention, self).__init__()
        self.__name__ = 'Attention'
        self.args = args
        self.quant_config = None
        self.parallel_config = None
        torch.set_default_dtype(get_dtype(precision=args['precision']))
        self.self_attn = self.set_type(_type, args)
        self.input_layernorm = None
        vllm_config = VllmConfig(quant_config=self.quant_config, parallel_config=self.parallel_config)
        set_current_vllm_config(vllm_config)

    def set_type(self, _type, args):
        self.state = {}
        self._type = _type
        self.configs = get_config(_type, args)
        config = DictConfig(args['shape'])

        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          163840)
        if 'has_rms_norm' in self.configs:
            self.input_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)
            self.state['desp'] = f"Norm"
            self.state['op'] = f"Norm"
        else:
            self.state['desp'] = ""
            self.state['op'] = ""
        self.quant_config = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
        self.parallel_config = ParallelConfig(pipeline_parallel_size=1, enable_expert_parallel=True)
        if 'use_mla' in self.configs:
            if 'use_fused_op' in self.configs and DeepseekV2MLAAttentionFusion:
                self.apply = DeepseekV2MLAAttentionFusion
                self.state['op'] += "DeepseekV2MLAAttentionFusion"
                self.state['desp'] += "DeepseekV2MLAAttentionFusion"
            else:
                self.apply = DeepseekV2MLAAttention
                self.state['op'] += "DeepseekV2MLAAttention"
                self.state['desp'] += "DeepseekV2MLAAttention"
        else:
            self.apply = DeepseekV2Attention
            self.state['op'] += "DeepseekV2Attention"
            self.state['desp'] += "DeepseekV2Attention"
        from vllm.attention.backends.flash_attn import FlashAttentionBackend
        return self.apply(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            # cache_config=cache_config,
            quant_config=self.quant_config,
            # attn_backend=FlashAttentionBackend(),
            prefix=f"layers.1.self_attn",
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
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor]=None,
        # actual_seqlen: Optional[int],
    ) -> torch.Tensor:
        # Self Attention
        global LocalVllmBackend
        hidden_states = hidden_states.clone()
        self.state['device'] = hidden_states.device.type
        self.state['hw_name'] = self.state['device']
        self.state['shape'] = self.state['shape'] = 'x'.join([str(dim) for dim in hidden_states.shape])
        M, N = hidden_states.shape

        attn_metadata = AttentionMetadata(
            num_prefills=1,
            num_prefill_tokens=M,
            num_decode_tokens=0,
            slot_mapping=torch.arange(0, M),
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
        )
        attn_metadata.is_profile_run = False
        attn_metadata.input_positions=torch.arange(0, M)
        # attn_metadata.context_chunk_workspace
        with set_forward_context(attn_metadata, get_current_vllm_config(), virtual_engine=0):
            self.self_attn.mla_attn.kv_cache[get_forward_context().virtual_engine].zero_()
            if 'has_rms_norm' in self.configs:
                if residual is None:
                    residual = hidden_states
                    hidden_states = self.input_layernorm(hidden_states)
                else:
                    hidden_states, residual = self.input_layernorm(
                        hidden_states, residual)
            # self.self_attn.process_weights_after_loading(dtype)
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
        if 'save_output_by_pt' in self.args and self.args['save_output_by_pt']:
            torch.save({'name': self.__name__, 'state': self.state, 'd': hidden_states,
                        'desp': self.state.get('desp')},
                    f'data/pt/vllm/{self.get_title(self.state)}.pt')
        LocalVllmBackend['n'] += 1
        print('out:', LocalVllmBackend['n'], hidden_states)
        if residual is None:
            return [hidden_states]
        else:
            return hidden_states, residual


def env(layer_type):
    pass


def get_config(layer_type, args):
    configs = args['config']
    return configs


def splite_test_case(args):
    case_dict = {'MLA': 'DeepseekV2MLAAttentionFusion'}
    if args['load_case_file']:
        sort_map = {'root': '.',
                    'layer': 'model.layers',
                    'pre': [],
                    'transformer': ['self_attn'],
                    'post': ['logits_processor'],}
        pt_loader.load(args['load_case_file'], case_dict.values(), map=sort_map)
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
                # for _init, _input in pt_loader.get(case_dict[main_type]):
                for _init, _input in pt_loader.get_by_module_cls(0, 4, 60, case_dict[main_type]), _iter=0):
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
            config = []
            module = Attention(module_type, args)
            print('model:', module.self_attn.state_dict().keys())
            if 'load' in args:
                data = torch.load(args['load'][0], weights_only=False)
                print('load:', data['init']['model']['_parameters'].keys())
                for i, k in  enumerate(data['init']['model']):
                    if k != '_parameters':
                        if k not in ['prefix', 'debug_layer_idx'] and data['init']['model'][k] != getattr(module.self_attn, k):
                            setattr(module.self_attn, k, data['init']['model'][k])
                        print('init:', i, k, data['init']['model'][k], getattr(module.self_attn, k))
                module.self_attn.load_state_dict(data['init']['model']['_parameters'], strict=False)
                print('load success:', data['init']['name'])
            else:
                data = torch.load(os.path.join(args['load_case_file'], 'attention_weight.pt'), weights_only=False)
                module.self_attn.load_state_dict(data)
        elif LOAD:
            module = torch.load('data/pt/launcher_attention_module.pt', weights_only=False)
            module.set_type(module_type, args)
        else:
            module = Attention(module_type, args)
            # torch.save(module, 'data/pt/launcher_attention_module.pt')
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
            dataset = torch.load('data/pt/launcher_attention_dataset.pt', weights_only=False)
        else:
            dataset = LocalDataset(shape_list, n_samples, dtype, args.task.acc._input_normal)
            # torch.save(dataset, 'data/pt/launcher_attention_dataset.pt')
    return dataset