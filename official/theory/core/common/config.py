from theory.core.loaders import loader
from theory.core.common.feature import Feature

class TheoryConfig(object):
    def __init__(self, name, cfg, user_cfg={}) -> None:
        self.name = name
        self.cfg = cfg
        self.user_cfg = user_cfg
        self.cfg_map = {}
        self.features = None
        self.hw_cfg = {
            'hw': 'A100PCIE',
            'sw': None,
            'n_nodes': 1,
            'n_device': 1
        }
        self.training_cfg = {
            'tensor_model_parallel_size': 1,
            'pipeline_model_parallel_size': 1,
            'virtual_pipeline_model_parallel_size': None,
            'data_parallel_size': 1,
            'context_parallel_size': 1,
            'expert_model_parallel_size': 1,
            'sequence_parallel': False,
            'gradient_accumulation_steps': 1,
            'micro_batch_size': 1,
            'global_batch_size': -1,
            'encoder_seq_length': -1,
            'seq_length': -1,
            'trained_samples': -1,
            'eval_samples': -1,
            'max_steps': -1,
            'kv_cache': -1,
            'pipeline_type': '1F1B',
        }
        self.inference_cfg = {
            'tensor_model_parallel_size': 1,
            'pipeline_model_parallel_size': 1,
            'virtual_pipeline_model_parallel_size': None,
            'data_parallel_size': 1,
            'global_batch_size': -1,
            'micro_batch_size': 1,
            'context_parallel_size': 1,
            'expert_model_parallel_size': 1,
            'sequence_parallel': False,
            'seq_length': -1,
            'kv_cache': -1,
            'task_type': None,
            'chunked_prefill': 10,
        }
        self.model_cfg = {
            'size': -1,
            'num_layers': -1,
            'hidden_size': -1,
            'ffn_hidden_size': -1,
            'num_attention_heads': -1,
            'num_hidden_dim': -1,
            'vocab_size': -1,
            'kv_channels': None,
            'kv_lora_rank': -1,
            'q_lora_rank': -1,
            'qk_nope_head_dim': -1,
            'qk_rope_head_dim': -1,
            'qk_head_dim': -1,
            'num_query_groups': -1,
            'num_experts': None,
            'untie_embeddings_and_output_weights': None,
            'gqa': None,
            'swiglu': None,
            'n_shared_experts': None,
            'n_routed_experts': None,
            'n_experts': None,
            'rms_norm': None,
            'normalization': None,
            'position_embedding_type': None,
            'attention_type': None,
            'moe_intermediate_size': None,
            'moe_router_topk': None,
            'n_expert_groups': None,
            'n_limited_groups': None,
            'first_k_dense_replace': -1,
            'num_layers_moe': -1,
            'routed_scaling_factor': None,
            'moe_layer_freq': None,
            'num_image_tokens': None,
            'text_len': None,
            'nsys_steps': -1,
            'nsys_ranks': [],
            'patch_size': -1,
            'input_size': [],
            'mlp_ratio': -1,
            'text_states_dim': -1,
            'per_group_quant_fp8': -1,
        }
        self.feature_cfg = {
            'megatron_amp_O2': None,
            'fsdp': None,
            'activations_checkpoint': None,
            'transformer_engine': None,
            'precision': 'fp32',
            'ub_tp_comm_overlap': None,
            'cpu_offloading': None,
            'use_distributed_optimizer': None,
            'fp8': None,
            'recompute_granularity': None,
            'transformer_impl': None,
            'fusion': None,
            'main_grads_dtype': 'bf16',
            'main_params_dtype': 'bf16',
            'exp_avg_dtype': 'fp32',
            'exp_avg_sq_dtype': 'fp32',
        }
        self.res = {
            'train_epoch_timing': -1,
            'train_step_timing': -1,
            'train_iter_timing': -1,
            'validation_epoch_timing': -1,
            'validation_step_timing': -1,
            'total_time': -1,
            'total_time_min': -1,
            'throughput': -1,
            'tps_per_device': -1,
            'eval_loss': -1,
            'train_loss': -1,
            'log_perplexity': -1,
            'nsys_steps': -1,
            'nsys_ranks': [],
        }
        self.metrics = {
            'total_tflops': 0,
            'model_params': 0,
            'estimated_params': 0,
        }
        self.cfg_mapping = {
            'env': 'hw_cfg',
            'model': 'model_cfg',
            'training': 'training_cfg',
            'inference': 'inference_cfg',
            'feature': 'feature_cfg',
            'modules': 'modules',
        }
        self.modules = {}
        self.graph = {}
        self.target = {}
        self.not_mapping = {}
        self.training = all('inference' not in k for k in self.cfg)
        self.run_type = 'training_cfg' if self.training else 'inference_cfg'
        self.__map__(self.run_type)
        if self.model_cfg['hidden_size'] == -1 and self.not_mapping.get('model'):
            self.load_model(self.not_mapping['model'])
        self.check()
        self.features = Feature(self)

    def __map__(self, run_type):
        mapped = []
        if not self.cfg:
            return
        for cfg in ['hw_cfg', run_type, 'model_cfg', 'feature_cfg', 'res']:
            _dict = getattr(self, cfg)
            for k in _dict:
                if k in self.cfg:
                    _dict[k] = self.cfg[k]
                    mapped.append(k)
        for key in self.cfg:
            if key in self.cfg_mapping:
                if isinstance(self.cfg[key], str) and key == 'model':
                    self.load_model(self.cfg[key])
                    continue
                for k in self.cfg[key]:
                    getattr(self, self.cfg_mapping[key])[k] = self.cfg[key][k]
        for k in self.cfg:
            if k not in mapped:
                self.not_mapping[k] = self.cfg[k]

    def update(self, cfg):
        for key in cfg:
            if key in self.cfg_mapping:
                for k in cfg[key]:
                    getattr(self, self.cfg_mapping[key])[k] = cfg[key][k]
        self.features = Feature(self)

    def load_model(self, model_name):
        model_obj = loader.load('model', cfg={'name': model_name})
        for k in self.model_cfg:
            if k in model_obj[model_name]:
                self.model_cfg[k] = model_obj[model_name][k]

    def set(self, attr, value):
        setattr(self, attr, value)

    def get(self, attr, key):
        return getattr(self, attr)[key]

    def tags(self, attr=None):
        if attr:
            return getattr(self.features, attr)
        return self.features.graph

    def shape(self, key):
        try:
            return self.features.shape[key]
        except:
            return self[key]

    def __getitem__(self, key):
        for cfg in ['hw_cfg', self.run_type, 'model_cfg', 'feature_cfg', 'res']:
            if key in getattr(self, cfg):
                return getattr(self, cfg)[key]

    def __setitem__(self, key, value):
        for cfg in ['hw_cfg', self.run_type, 'model_cfg', 'feature_cfg', 'res']:
            if key in getattr(self, cfg):
                getattr(self, cfg)[key] = value
                return

    def __iter__(self):
        for cfg in ['hw_cfg', self.run_type, 'model_cfg', 'feature_cfg']:
            _dict = getattr(self, cfg)
            for k in _dict:
                if _dict[k] not in [-1, None, 'None'] or k == 'kv_cache':
                    yield k, _dict[k]

    def print(self):
        print('*'*50, self.name ,'*'*50)
        for cfg in ['hw_cfg', self.run_type, 'model_cfg', 'feature_cfg', 'res', 'modules']:
            _dict = getattr(self, cfg)
            for k in _dict:
                if _dict[k] not in [-1, None, 'None']:
                    print('\t', cfg, ':', k, _dict[k])

    def size2str(self, size):
        if size % (1024*1024) == 0:
            return f'{int(size/1024/1024)}M'
        elif size % 1024 == 0:
            return f'{int(size/1024)}K'
        else:
            return f'{size}'

    def update_test_id(self, cfg):
        feature_names = {
            'tensor_model_parallel_size': 'T',
            'context_parallel_size': 'C',
            'expert_model_parallel_size': 'E',
            'pipeline_model_parallel_size': 'P',
            'global_batch_size': 'GB',
            'micro_batch_size': 'MB',
        }
        name = ''
        for c in cfg:
            for k, v in cfg[c].items():
                if self[k] != v:
                    self[k] = v
                    name += feature_names[k] + str(v)
                    print(k, self[k], v, name)
        return name

    def check(self, hw=None):
        self.name = hw if hw else self.hw_cfg['hw']
        parallel_id = ''
        if getattr(self, self.run_type)['tensor_model_parallel_size'] > 1:
            if not getattr(self, self.run_type)['sequence_parallel']:
                getattr(self, self.run_type)['tensor_parallel'] = True
            parallel_id += f"T{getattr(self, self.run_type)['tensor_model_parallel_size']}"
        if getattr(self, self.run_type)['context_parallel_size'] > 1:
            getattr(self, self.run_type)['context_parallel'] = True
            parallel_id += f"C{getattr(self, self.run_type)['context_parallel_size']}"
        if getattr(self, self.run_type)['expert_model_parallel_size'] > 1:
            getattr(self, self.run_type)['expert_parallel'] = True
            parallel_id += f"E{getattr(self, self.run_type)['expert_model_parallel_size']}"
        if getattr(self, self.run_type)['pipeline_model_parallel_size'] > 1:
            parallel_id += f"P{getattr(self, self.run_type)['pipeline_model_parallel_size']}"
        if parallel_id:
            self.name += '_' + parallel_id
        if getattr(self, self.run_type)['global_batch_size'] != -1:
            self.name += f"_GB{getattr(self, self.run_type)['global_batch_size']}MB{getattr(self, self.run_type)['micro_batch_size']}"
        else:
            self.name += f"_MB{getattr(self, self.run_type)['micro_batch_size']}"
        self.name += 'fp8' if self.feature_cfg['fp8'] else self.feature_cfg['precision']
        self.name += '_' + self.size2str(getattr(self, self.run_type)['seq_length'])
        if self.model_cfg['qk_nope_head_dim'] != -1 and self.model_cfg['qk_rope_head_dim'] != -1:
            self.model_cfg['qk_head_dim'] = self.model_cfg['qk_nope_head_dim'] + self.model_cfg['qk_rope_head_dim']
        if self.model_cfg['num_query_groups'] != -1:
            self.model_cfg['gqa'] = True
            self.model_cfg['num_query_groups'] *= (2 + self.model_cfg['num_attention_heads'] / self.model_cfg['num_query_groups'])
            self.model_cfg['position_embedding_type'] = None
        if self.model_cfg['first_k_dense_replace'] != -1:
            self.model_cfg['num_layers_moe'] = self.model_cfg['num_layers'] - self.model_cfg['first_k_dense_replace']
        if self.feature_cfg['transformer_impl'] == 'vllm' and self.feature_cfg['fp8']:
            self.feature_cfg['transformer_impl'] = 'vllm_fp8'
        if self.inference_cfg['chunked_prefill']:
            self.inference_cfg['pipeline_type'] = 'chunked_prefill'