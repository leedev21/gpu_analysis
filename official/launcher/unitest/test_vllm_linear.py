# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for cutlass kernels

Run `pytest tests/kernels/test_cutlass.py`.
"""
import random

import pytest
import torch

from utils import baseline_scaled_mm, opcheck, to_fp8, to_int8
from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.utils import cdiv

MNK_FACTORS = [
    (1, 256, 128),
    (1, 16384, 1024),
    (1, 24576, 496),
    (16, 256, 496),
    (16, 16384, 128),
    (16, 24576, 4096),
    (32, 8192, 4096),
    (32, 16384, 4096),
    (33, 1024, 1024),
    (33, 8192, 128),
    (64, 2048, 496),
    (64, 16384, 1024),
    (100, 8192, 496),
    (128, 32768, 4096),
    (256, 4096, 4096),
    (512, 256, 1024),
    (512, 8192, 4096),
    (512, 16384, 128),
    (512, 24576, 128),
]

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)
]

# -1 means full extent in that dimension
TENSORWISE_GROUP_SHAPE = (-1, -1)
PER_TOKEN_GROUP_SHAPE = (1, -1)
PER_OUT_CH_GROUP_SHAPE = (-1, 1)

capability = current_platform.get_device_capability()
capability = capability[0] * 10 + capability[1]


def rand_int8(shape: tuple, device: str = "cuda"):
    return to_int8(torch.rand(shape, device=device) * 255 - 128)


def group_scale_helper(shape, group_shape):
    return [shape[i] if s < 0 else s for i, s in enumerate(group_shape)]


def scale_shape(shape, group_shape):
    assert len(shape) == len(group_shape)
    group_shape = group_scale_helper(shape, group_shape)
    return tuple(
        cdiv(shape[i], group_shape[i]) for i in range(len(group_shape)))


def cutlass_fp8_gemm_helper(m: int,
                            n: int,
                            k: int,
                            a_scale_group_shape: tuple,
                            b_scale_group_shape: tuple,
                            use_bias: bool,
                            out_dtype: type[torch.dtype] = torch.bfloat16,
                            device: str = "cuda"):
    # Test for a cutlass kernel with per-token activation quantization
    # and per-output channel weight quantization.
    a = to_fp8(torch.randn((m, k), device=device))
    b = to_fp8(torch.randn((n, k), device=device).t())

    a_scales_shape = scale_shape(a.shape, a_scale_group_shape)
    b_scales_shape = scale_shape(b.shape, b_scale_group_shape)

    scale_a = (torch.randn(a_scales_shape, device=device, dtype=torch.float32))
    scale_b = (torch.randn(b_scales_shape, device=device, dtype=torch.float32))

    # make scales M-major for blockwise quant, doesn't affect 1D scales
    scale_a = scale_a.t().contiguous().t()
    # make scales K-major for blockwise quant, doesn't affect 1D scales
    scale_b = scale_b.t().contiguous().t()

    if use_bias:
        bias = torch.rand((n, ), device=device, dtype=out_dtype) * 10
    else:
        bias = None

    out = ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    baseline = baseline_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    torch.testing.assert_close(out, baseline, rtol=1e-2, atol=1.5e-1)

    opcheck(torch.ops._C.cutlass_scaled_mm,
            (out, a, b, scale_a, scale_b, bias))


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("a_scale_group_shape,b_scale_group_shape",
                         [((1, 128), (128, 128))])
@pytest.mark.parametrize("use_bias", [False])
@pytest.mark.skipif(not current_platform.has_device_capability(90),
                    reason="FP8 blockwise is not supported on this GPU type.")
def test_cutlass_fp8_blockwise_scale_gemm(m: int, n: int, k: int,
                                          a_scale_group_shape,
                                          b_scale_group_shape, use_bias: bool):
    if k % b_scale_group_shape[0] != 0 or n % b_scale_group_shape[1] != 0:
        return
    if m % a_scale_group_shape[0] != 0 or k % a_scale_group_shape[1] != 0:
        return
    if m % 4 != 0 and current_platform.has_device_capability(100):
        return
    cutlass_fp8_gemm_helper(m, n, k, a_scale_group_shape, b_scale_group_shape,
                            use_bias)