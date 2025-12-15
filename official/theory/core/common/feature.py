class Feature(object):
    def __init__(self, config):
        self.name = config.name
        self.type = None
        self.apply_func = None
        self.feature_map = {
            'shape':{
                'micro_batch_size': 'B',
                'num_attention_heads': 'H',
                'seq_length': 'M',
                'num_hidden_dim': 'D',
                'hidden_size': 'N',
                'ffn_hidden_size': 'F',
                'vocab_size': 'E',
                'kv_cache': 'KV',
                'params_size': 'P',
                "patch_size": 'P',
                'num_layers': 'L',
                'tensor_model_parallel_size': 'T',
                'context_parallel_size': 'C',
                'expert_model_parallel_size': 'EP',
                'num_image_tokens': 'IM',
                'image_size': 'IS',
                'text_len': 'TM',
                'moe_ffn_hidden_size': 'MF',
                'num_query_groups': 'NQ',
                'kv_lora_rank': 'KVR',
                'q_lora_rank': 'QR',
                'qk_nope_head_dim': 'QKN',
                'qk_rope_head_dim': 'QKR',
                'qk_head_dim': 'DQK',
                'n_experts': 'NE',
                'n_shared_experts': 'NSE',
                'n_routed_experts': 'NRE',
                'moe_router_topk': 'MRT',
                'n_expert_groups': 'NEG',
                'n_limited_groups': 'NLG',
                'moe_layer_freq': 'MLF',
                'first_k_dense_replace': 'MKD',
                'num_layers_moe': 'MLE',
                'routed_scaling_factor': 'MRS',
                'per_group_quant_fp8': 'PGQ',
            },
            'graph':{
                'rope': 'Attention',
                'mla': 'Attention',
                'gqa': 'Attention',
                'te': 'Module',
                'apex': 'Module',
                'vllm': 'Module',
                'grouped_topk': 'Router',
                'fused_topk': 'Router',
                'rms_norm': 'LayerNorm',
                'silu': 'Activation',
                'gelu': 'Activation',
                'zero1': 'Optimizer',
                'zero2': 'Optimizer',
                'zero3': 'Optimizer',
                'tensor_parallel': 'Module',
                'sequence_parallel': 'Module',
                'context_parallel': 'Attention',
                'expert_parallel': 'Module',
                'i2v': 'Attention',
                't2v': 'Attention',
                'recompute': 'Module', # enable in graph, add forward before backward
                'vllm_fp8': 'Gemm',
            },
            # 'parallel':{
            #     'tensor_parallel': 'Module',
            #     'sequence_parallel': 'Module',
            #     'context_parallel': 'Attention',
            # },
            'runner':{
                'decoding': 'Model',
                'precision': 'Model',
                'megatron_amp_O2': 'Optimizer',
                'megatron_amp_O1': 'Optimizer',
                'fsdp': 'Module',
                'global_batch_size': 'Model',
                'gradient_accumulation_steps': 'Model',
                'pipeline_model_parallel_size': 'Model',
                'virtual_pipeline_model_parallel_size': 'Model',
            },
            'summary':{
                'activations_checkpoint': 'Module',
                'tp_overlap': 'Comm',
                'dp_overlap': 'Comm',
                'ep_overlap': 'Comm',
                'pp_overlap': 'Comm',
                'cuda_graph': 'Comm',
                'optim_overlap': 'Comm',
                'data_parallel_size': 'Model',
                'zero_bubble': 'Model',
                'dual_pipe': 'Model',
                # 'expert_model_parallel_size': 'Module',
                # 'n_shared_experts': 'Module',
                # 'n_routed_experts': 'Module',
                # 'moe_router_topk': 'Module',
                'moe_layer_freq': 'Module',
                'trained_samples': 'Runner',
                'max_steps': 'Runner',
            },
            'set':{
                'swiglu': ['silu'],
                'moe_intermediate_size': ['moe_ffn_hidden_size'],
                'use_distributed_optimizer': ['zero1'],
                'ub_tp_comm_overlap': ['tp_overlap'],
                'transformer_engine': ['te'],
                'fp8': {'precision': 'fp8'},
                'overlap': ['tp_overlap', 'dp_overlap', 'ep_overlap', 'pp_overlap', 'optim_overlap'],
            },
            'mapping':{
                'normalization': {
                    'RMSNorm': 'rms_norm',
                    'rmsnorm': 'rms_norm',
                    'LayerNorm': 'base'
                },
                'position_embedding_type': {
                    'rope': 'rope'
                },
                'transformer_impl': {
                    'transformer_engine': 'te',
                    'vllm': 'vllm',
                    'vllm_fp8': ['vllm_fp8', 'vllm'],
                    'deepseek': 'deepseek'
                },
                'attention_type': {
                    'mla': 'mla',
                    'multihead': 'base',
                },
                'task_type': {
                    'i2v': 'i2v',
                    't2v': 't2v',
                }
            }
        }
        self.shape = {}
        self.graph = {}
        self.parallel = {}
        self.runner = {}
        self.summary = {}
        self.__map__(config)

    def __map__(self, config):
        feature_map = dict()
        def update(k, v):
            if feature_map[k]['g'] == 'set':
                if isinstance(feature_map[k]['v'], list):
                    for tag in feature_map[k]['v']:
                        update(tag, v)
                if isinstance(feature_map[k]['v'], dict):
                    if v:
                        for key, value in feature_map[k]['v'].items():
                            update(key, value)
            elif feature_map[k]['g'] == 'mapping':
                if v in feature_map[k]['v'] and feature_map[k]['v'][v] != 'base':
                    if isinstance(feature_map[k]['v'][v], list):
                        for tag in feature_map[k]['v'][v]:
                            update(tag, True)
                    else:
                        update(feature_map[k]['v'][v], True)
            elif feature_map[k]['g'] == 'shape':
                getattr(self, feature_map[k]['g'])[feature_map[k]['v']] = v
            else:
                getattr(self, feature_map[k]['g'])[k] = v
        for group in self.feature_map:
            for k in self.feature_map[group]:
                feature_map[k] = {'g': group, 'v': self.feature_map[group][k]}
        for k, v in config:
            if k in feature_map:
                update(k, v)
        self.print()

    def print(self):
        for cfg in ['shape', 'graph', 'parallel', 'runner', 'summary']:
            _dict = getattr(self, cfg)
            for k in _dict:
                if _dict[k] not in [-1, None, 'None']:
                    print('\t', cfg, ':', k, _dict[k])
        # exit()
