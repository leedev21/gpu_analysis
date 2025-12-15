# SPDX-License-Identifier: Apache-2.0
"""
Test:

* Tests for MultiHeadAttention layer
"""
from unittest.mock import patch
import os
try:
    from torchtrace.torchtrace import set_torchtrace, update
    update('customer_op')
except:
    pass
import pytest
import torch

from vllm.attention.layer import Attention
from vllm.attention.selector import _Backend, _cached_get_attn_backend
from vllm.platforms import current_platform
from vllm.platforms.cpu import CpuPlatform
from vllm.platforms.cuda import CudaPlatform
from vllm.platforms.rocm import RocmPlatform


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear lru cache to ensure each test case runs without caching.
    """
    _cached_get_attn_backend.cache_clear()


# @pytest.mark.parametrize("device", ["cpu", "hip", "cuda"])
# def test_mha_attn_platform(device: str):
#     """
#     Test the attention selector between different platform and device.
#     """
#     torch.set_default_dtype(torch.float16)

#     if device == "cpu":
#         with patch("vllm.attention.selector.current_platform", CpuPlatform()):
#             attn = Attention(16, 64, scale=1)
#             assert attn.attn_backend == _Backend.TORCH_SDPA
#     elif device == "hip":
#         with patch("vllm.attention.selector.current_platform", RocmPlatform()):
#             attn = Attention(16, 64, scale=1)
#             assert attn.attn_backend == _Backend.TORCH_SDPA
#     else:
#         with patch("vllm.attention.selector.current_platform", CudaPlatform()):
#             attn = Attention(16, 64, scale=1)
#             assert attn.attn_backend == _Backend.XFORMERS

#         with patch("vllm.attention.selector.current_platform", CudaPlatform()):
#             attn = Attention(16, 72, scale=1)
#             assert attn.attn_backend == _Backend.XFORMERS


def ref_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Native implementation of scaled dot product attention without mask:
    - query, key, value: [batch_size, seq_len, num_heads, head_size]
    - attn_mask: [batch_size, seq_len, seq_len]
    """
    query, key, value = (x.transpose(1, 2) for x in (query, key, value))
    attn_weights = scale * torch.matmul(query, key.transpose(2, 3))
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.matmul(attn_weights, value).transpose(1, 2)
    return out

if set_torchtrace and os.getenv('RUN_TYPE', '') == 'trace':
    set_torchtrace(torch_dispatch_trace=True, torch_api_trace=True, save_pt=False, sync_mode=True)

BATCH_SIZES = [1, 16]
SEQ_LENS = [1]
NUM_HEADS = [1, 16]
NUM_KV_HEADS = [1]
HEAD_SIZES = [64, 80]
# flshattF and tritonflashattF supported: {torch.float16, torch.bfloat16}
DTYPES = [torch.half, torch.bfloat16, torch.float]
CUDA_DEVICES = ["cuda"]

from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.linear import (ColumnParallelLinear, RowParallelLinear)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.attention.selector import (_Backend, _cached_get_attn_backend,
                                     global_force_attn_backend_context_manager)
from vllm.attention import AttentionType
from utils import make_backend, init_test_distributed_environment
from vllm.distributed.parallel_state import (get_tensor_model_parallel_group,
                                             get_tp_group, graph_capture)

init_test_distributed_environment(1, 1, 0, '1234')
group = get_tensor_model_parallel_group().device_group

LIST_ENC_DEC_SUPPORTED_BACKENDS = [_Backend.XFORMERS, _Backend.FLASH_ATTN]
kv_lora_rank = 512
q_lora_rank = 1536
qk_nope_head_dim = 128
qk_rope_head_dim = 64
qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
v_head_dim = 64
hidden_size = 7168
num_attention_heads = 128
rope_scaling = {
    'rope_type': 'deepseek_yarn',
}
rotary_emb = get_rope(qk_rope_head_dim,
                    rotary_dim=qk_rope_head_dim,
                    max_position=8192,
                    base=10000,
                    rope_scaling=None,
                    is_neox_style=False)

def make_test_metadata(attn_backend, batch_size, _graph_seq_lens, max_seq_len, input_positions):
    attn_metadata = attn_backend.make_metadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=batch_size,
        slot_mapping=None,
        multi_modal_placeholder_index_maps=None,
        enable_kv_scales_calculation=True,
        seq_lens=None,
        seq_lens_tensor=_graph_seq_lens[:batch_size],
        max_query_len=1,
        max_decode_query_len=1,
        max_prefill_seq_len=0,
        max_decode_seq_len=max_seq_len,
        query_start_loc=None,
        seq_start_loc=None,
        context_lens_tensor=None,
        block_tables=None,
        use_cuda_graph=True,
        # input_positions=input_positions
    )
    return attn_metadata


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_kv_heads", NUM_KV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@pytest.mark.parametrize("attn_backend", LIST_ENC_DEC_SUPPORTED_BACKENDS)
def test_attn_forward(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: str,
    attn_backend,
):
    current_platform.seed_everything(0)
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)

    # q = torch.randn(batch_size, seq_len, num_heads * head_size)
    q = torch.randn(batch_size, seq_len, q_lora_rank)
    k = torch.randn(batch_size, seq_len, num_kv_heads * head_size)
    v = torch.randn(batch_size, seq_len, num_kv_heads * head_size)
    kv_cache = torch.randn(seq_len, 1024 * head_size)
    print('inputs:', q.shape, k.shape, kv_cache.shape)
    q_b_proj = ColumnParallelLinear(q_lora_rank,
                                    num_attention_heads *
                                    qk_head_dim,
                                    bias=False,)
                                    # quant_config=quant_config,
                                    # prefix=f"{prefix}.q_b_proj")
    kv_b_proj = ColumnParallelLinear(
                kv_lora_rank,
                num_attention_heads * (qk_nope_head_dim + v_head_dim),
                bias=False,)
                # quant_config=quant_config,
                # prefix=f"{prefix}.kv_b_proj")
    o_proj = RowParallelLinear(num_attention_heads * v_head_dim,
                                hidden_size,
                                bias=False,)
                                # quant_config=quant_config,
                                # prefix=f"{prefix}.o_proj")
    
    vllm_config = VllmConfig()
    
    with set_current_vllm_config(vllm_config):
        _graph_seq_lens = torch.ones(batch_size,
                                    dtype=torch.int32,
                                    device=device)
        with global_force_attn_backend_context_manager(attn_backend):
            attn_backend_obj = make_backend(attn_backend.name)
            
            input_positions = torch.ones(seq_len, hidden_size, dtype=torch.long)
            decphase_attn_metadata = make_test_metadata(
                attn_backend_obj,
                batch_size,
                _graph_seq_lens,
                seq_len,
                input_positions)

            scale = 1.0 / head_size**0.5
            attn = Attention(num_heads,
                                    head_size,
                                    scale=scale,
                                    use_mla=True,
                                    q_lora_rank=q_lora_rank,
                                    kv_lora_rank=kv_lora_rank,
                                    qk_nope_head_dim=qk_nope_head_dim,
                                    qk_rope_head_dim=qk_rope_head_dim,
                                    qk_head_dim=qk_head_dim,
                                    v_head_dim=v_head_dim,
                                    # rotary_emb=rotary_emb,
                                    # q_proj=q_b_proj,
                                    kv_b_proj=kv_b_proj,
                                    # o_proj=o_proj,
                                    prefix=f"{AttentionType.DECODER}",
                                    num_kv_heads=num_kv_heads)
            with set_forward_context(decphase_attn_metadata, vllm_config):
                attn.process_weights_after_loading(dtype)
                output = attn(q, k, v, kv_cache)#, decphase_attn_metadata)
test_attn_forward(1, 128, 128, 64, 56, torch.bfloat16, 'cuda', _Backend.TRITON_MLA)
