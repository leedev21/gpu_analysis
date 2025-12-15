# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""
import pytest
# try:
from torchtrace.torchtrace import set_torchtrace, update
update('customer_op')
# except:
#     pass
import os

import torch
from transformers import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

import vllm.model_executor.layers.fused_moe  # noqa
from utils import (compute_max_diff, opcheck, stack_and_dev,
                                 torch_moe, torch_moe_single)
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, moe_align_block_size)
from vllm.model_executor.layers.fused_moe.moe_torch_iterative import (
    fused_moe as iterative_moe)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
    marlin_quantize)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    quantize_weights)
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

import atexit

if set_torchtrace and os.getenv('RUN_TYPE', '') == 'trace':
    # update('filter', 'op_trace', {'op': ['aten::index'], 'max': 2})
    set_torchtrace(torch_dispatch_trace=True, torch_api_trace=True, save_pt=True, sync_mode=True)
    M = [64]
    N = [128]
    K = [511]
    NUM_EXPERTS = [8]
    EP_SIZE = [1, 4]
    TOP_KS = [2]
else:
    M = [1, 33, 64, 222, 1024 * 128]
    N = [128, 1024, 2048]
    K = [128, 511, 1024]
    NUM_EXPERTS = [8, 64]
    EP_SIZE = [1, 4]
    TOP_KS = [2, 6]


# @pytest.mark.parametrize("m", M)
# @pytest.mark.parametrize("n", N)
# @pytest.mark.parametrize("k", K)
# @pytest.mark.parametrize("e", NUM_EXPERTS)
# @pytest.mark.parametrize("topk", TOP_KS)
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# def test_fused_moe(
#     m: int,
#     n: int,
#     k: int,
#     e: int,
#     topk: int,
#     dtype: torch.dtype,
# ):
#     a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
#     w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
#     w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

#     score = torch.randn((m, e), device="cuda", dtype=dtype)
#     triton_output = fused_moe(a, w1, w2, score, topk, renormalize=False)
#     torch_output = torch_moe(a, w1, w2, score, topk)
#     torch.testing.assert_close(triton_output, torch_output, atol=2e-2, rtol=0)
#     iterative_output = iterative_moe(a, w1, w2, score, topk, renormalize=False)
#     torch.testing.assert_close(iterative_output,
#                                torch_output,
#                                atol=2e-2,
#                                rtol=0)

# test_fused_moe(64, 128, 511, 4, 2, torch.bfloat16)

@pytest.mark.parametrize("m", M)
@pytest.mark.parametrize("n", N)
@pytest.mark.parametrize("k", K)
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("ep_size", EP_SIZE)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_moe(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    ep_size: int,
    dtype: torch.dtype,
):
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
    atol = 1e-2 if dtype == torch.bfloat16 else 1e-3

    triton_output = fused_moe(a,
                              w1,
                              w2,
                              score,
                              topk,
                              global_num_experts=e,
                              expert_map=e_map,
                              renormalize=False)
    torch_output = torch_moe(a, w1, w2, score, topk, e_map)
    torch.testing.assert_close(triton_output, torch_output, atol=atol, rtol=0)
    iterative_output = iterative_moe(a,
                                     w1,
                                     w2,
                                     score,
                                     topk,
                                     global_num_experts=e,
                                     expert_map=e_map,
                                     renormalize=False)
    torch.testing.assert_close(iterative_output,
                               torch_output,
                               atol=atol,
                               rtol=0)


test_fused_moe(64, 128, 511, 4, 2, 4, torch.bfloat16)
