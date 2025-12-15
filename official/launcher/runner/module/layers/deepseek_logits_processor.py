import torch
from torch import nn
import os, json
from typing import Optional, Union
from .quant import native_per_token_group_quant_fp8
from .native import RMSNorm
try:
    import vllm._custom_ops as ops
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

except ImportError:
    print('vllm not found')
    ops = None

from .base import RandomDataset, get_dtype, generate_tensor_with_stats
from copy import deepcopy
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig


LOAD = False


class LocalDataset(RandomDataset):
    def new(self, shape_list, dtype):
        for batch_size, seq_length, hidden_size in shape_list:
            if self.distribution:
                hidden_states = generate_tensor_with_stats([batch_size*seq_length, hidden_size], dtype=dtype, device='cpu', args=self.distribution, seed=1)
            else:
                scale = 1 / (hidden_size)
                hidden_states = torch.randn(batch_size*seq_length, hidden_size, dtype=dtype) * scale

            self.data.append((hidden_states, ))


class LocalLogitsProcessor(nn.Module):
    def __init__(self, _type, args):
        super(FusedMoe, self).__init__()
        self.__name__ = 'FusedMoe'
        self.args = args
        self.routed_scaling_factor = None
        self.log2phy = None
        self.prior_expert_map = None
        dtype = get_dtype(precision=args['precision'])
        self.logits_processor = self.set_type(_type, args).to(dtype)
        self.post_attention_layernorm = None

    def set_type(self, _type, args):
        self.state = {}
        self._type = _type
        self.configs = get_config(_type, args)
        prefix = 'module.0'
        config = DictConfig(args['shape'])
        print('check1', args)
        print('check2', self.configs)

        if 'has_rms_norm' in self.configs:
            self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                        eps=config.rms_norm_eps)
            self.state['desp'] = f"Norm"
        else:
            self.state['desp'] = ""

        self.state['desp'] += f"LogitsProcessor"
        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size, quant_config=quant_config
        )
        return LogitsProcessor(config.vocab_size)

    @staticmethod
    def get_title(state):
        name = f"{state['op']}"
        for k in ['desp', 'shape', 'dtype', 'hw_name']:
            if k in state:
                name += f"_{state[k]}"
        return name

    def forward(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        self.state['device'] = hidden_states.device.type
        self.state['hw_name'] = self.state['device']
        self.state['shape'] = self.state['shape'] = 'x'.join([str(dim) for dim in hidden_states.shape])
        if 'has_rms_norm' in self.configs:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual)

        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits


def env(layer_type):
    pass


def get_config(layer_type, args):
    configs = args['config']
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
                    'init': {'shape': init,
                             'config': configs}
                    }


module = None
def get_module(module_type, args):
    global module
    if not module:
        if args['load_case_file']:
            module = LocalLogitsProcessor(module_type, args)
            module.module.weight.data = torch.load(os.path.join(args['load_case_file'], 'logits_processor_weight.pt'), weights_only=False)['weight']
        elif LOAD:
            module = torch.load('data/pt/launcher_logits_processor_module.pt', weights_only=False)
            module.set_type(module_type, args)
        else:
            module = LocalLogitsProcessor(module_type, args)
            # torch.save(module, 'data/pt/launcher_logits_processor_module.pt')
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
            dataset = torch.load('data/pt/launcher_logits_processor_dataset.pt', weights_only=False)
        else:
            dataset = LocalDataset(shape_list, n_samples, dtype, args.task.acc._input_normal)
            torch.save(dataset, 'data/pt/launcher_logits_processor_dataset.pt')
    return dataset