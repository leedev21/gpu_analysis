import os
import torch
import torch.nn as nn
import math
from importlib import import_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from .base import RandomDataset


def get_cls(path, module_name):
    module = import_module(path)
    if module and hasattr(module, module_name):
        return getattr(module, module_name)
    return None


scaling = 4
context_parallel_size = 2
tensor_model_parallel_size = int(8/scaling)
if scaling == 8:
    sequence_parallel = False
else:
    sequence_parallel = True


config = TransformerConfig(tensor_model_parallel_size=tensor_model_parallel_size, context_parallel_size=context_parallel_size,
            pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None,
            sequence_parallel=sequence_parallel, expert_model_parallel_size=1, perform_initialization=True,
            use_cpu_initialization=False, fp16=True, bf16=False, params_dtype=torch.float16,
            timers=None, gradient_accumulation_fusion=True, async_tensor_model_parallel_allreduce=False,
            tp_comm_overlap=False, tp_comm_split_ag=True, tp_comm_atomic_ag=False, tp_comm_split_rs=True,
            tp_comm_atomic_rs=False, tp_comm_bulk_wgrad=True, tp_comm_bulk_dgrad=True,
            finalize_model_grads_func=None, pipeline_dtype=torch.float16,
            # grad_scale_func=<bound method GradScaler.scale of <torch.cuda.amp.grad_scaler.GradScaler object at 0x7fea3f1dbfa0>>,
            enable_autocast=False, autocast_dtype=torch.float16, variable_seq_lengths=False,
            num_microbatches_with_partial_activation_checkpoints=None, overlap_p2p_comm=False,
            batch_p2p_comm=True, batch_p2p_sync=True, use_ring_exchange_p2p=False,
            deallocate_pipeline_outputs=True, no_sync_func=None, grad_sync_func=None,
            param_sync_func=None, pipeline_model_parallel_split_rank=None, cpu_offloading=False,
            cpu_offloading_num_layers=0, _cpu_offloading_context=None, cpu_offloading_activations=True,
            cpu_offloading_weights=True, barrier_with_L1_time=True, num_layers=1, hidden_size=12288/scaling,
            num_attention_heads=96/scaling, num_query_groups=96/scaling, ffn_hidden_size=49152, kv_channels=128,
            hidden_dropout=0.1, attention_dropout=0.1, fp32_residual_connection=False,
            apply_residual_connection_post_layernorm=False, layernorm_epsilon=1e-05,
            layernorm_zero_centered_gamma=False, add_bias_linear=True, add_qkv_bias=False,
            gated_linear_unit=False, num_moe_experts=None, rotary_interleaved=False, window_size=None,
            init_method_std=0.006, apply_query_key_layer_scaling=False, attention_softmax_in_fp32=True,
            # apply_query_key_layer_scaling=True
            bias_activation_fusion=True, masked_softmax_fusion=True, persist_layer_norm=True,
            memory_efficient_layer_norm=False, bias_dropout_fusion=True, apply_rope_fusion=False,
            recompute_granularity=None, recompute_method=None, recompute_num_layers=None,
            distribute_saved_activations=False, fp8=None, fp8_margin=0, fp8_interval=1,
            fp8_amax_history_len=1024, fp8_amax_compute_algo='max', fp8_wgrad=True,
            clone_scatter_output_in_embedding=True, normalization='LayerNorm',
            moe_router_load_balancing_type='aux_loss', moe_router_topk=2, moe_grouped_gemm=False,
            moe_aux_loss_coeff=0, moe_z_loss_coeff=None, moe_input_jitter_eps=None,
            moe_token_dropping=False)


def env(layer_type):
    config = {
        'TE unfused': {'NVTE_FLASH_ATTN': 0, 'NVTE_FUSED_ATTN': 0, 'NVTE_MASKED_SOFTMAX_FUSION': 0},
        'TE fused': {'NVTE_FLASH_ATTN': 0, 'NVTE_FUSED_ATTN': 1, 'NVTE_MASKED_SOFTMAX_FUSION': 1},
        'TE_flash-attn': {'NVTE_FLASH_ATTN': 1, 'NVTE_FUSED_ATTN': 0, 'NVTE_MASKED_SOFTMAX_FUSION': 0},
        'TE fuse softmax': {'NVTE_FLASH_ATTN': 0, 'NVTE_FUSED_ATTN': 0, 'NVTE_MASKED_SOFTMAX_FUSION': 1},
        'Megatron': {},
        'Torch': {},
    }
    os.environ['API_USER'] = 'False'
    for k, v in config[layer_type].items():
        os.environ[k] = str(v)
        print('Env Set:', k, v)


class TorchAttention(nn.Module):
    def __init__(self, config, layer_number, attn_mask_type, attention_type):
        super().__init__()
        self.head_dim = 128
        self.mask = None
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def load_mask(self, query, start_pos=0):
        bsz, head, seqlen, kv_channel = query.shape
        mask = torch.full((seqlen, seqlen), float("-inf"), device=query.device)
        mask = torch.triu(mask, diagonal=1)

        # When performing key-value caching, we compute the attention scores
        # only for the new sequence. Thus, the matrix of scores is of size
        # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
        # j > cache_len + i, since row i corresponds to token cache_len + i.
        self.mask = torch.hstack(
            [torch.zeros((seqlen, start_pos), device=query.device), mask]
        ).type_as(query)

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=AttnMaskType.causal,
                packed_seq_params=None):
        seqlen, bsz, head, kv_channel = query.shape
        query = query.view(bsz, seqlen, head, kv_channel)
        key = key.view(bsz, seqlen, head, kv_channel)
        value = value.view(bsz, seqlen, head, kv_channel)
        query = query.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        key = key.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        else:
            if self.mask is None:
                self.load_mask(query)
            scores = scores + self.mask
        scores = nn.functional.softmax(scores.float(), dim=-1).type_as(query)
        scores = self.attention_dropout(scores)
        output = torch.matmul(scores, value)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return output


class GPTDotProductAttention(nn.Module):
    def __init__(self, attention_type, args):
        super(GPTDotProductAttention, self).__init__()
        if attention_type == 'Megatron':
            model = DotProductAttention
        elif 'TE' in attention_type:
            model = get_cls('megatron.core.transformer.custom_layers.transformer_engine', 'TEDotProductAttention')
        elif attention_type == 'Torch':
            model = TorchAttention
        else:
            print('No this kind of core attention:', attention_type)
            exit(0)
        self.core_attention = model(
                    config=config,
                    layer_number=1,
                    attn_mask_type=AttnMaskType.causal,
                    attention_type="self"
            )

    def forward(self, query, key, value, attention_mask=None, attn_mask_type=AttnMaskType.causal,
                packed_seq_params=None):
        output = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
        )
        return output


def splite_test_case(args):
    for typ in args.module.layer.type:
        for seq_length in args.module.layer.seq_length:
            yield {'type': typ,
                   'seq_length': int(seq_length),
                   'init': {}
                   }


def get_module(module_type, args):
    return GPTDotProductAttention(module_type, args)


data_loader = RandomDataset