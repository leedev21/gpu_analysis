# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from __future__ import annotations

from collections.abc import Iterable
import io
import math
from typing import Optional

import pytest
import torch

import transformer_engine
import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch.ops._common import is_float8_tensor
from transformer_engine.pytorch.ops.fused import (
    BackwardLinearAdd,
    ForwardLinearBiasActivation,
    ForwardLinearBiasAdd,
)
try:
    from transformer_engine.pytorch.tensor import QuantizedTensor
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor, Float8Quantizer
except:
    Float8Quantizer = None
from transformer_engine.pytorch.utils import is_bf16_compatible
import transformer_engine_torch as tex

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()

try:
    mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()
except:
    pass

# Supported data types
_dtypes: list[torch.dtype] = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    _dtypes.append(torch.bfloat16)

# Supported devices
_devices: list[torch.device] = [torch.device("cpu"), torch.device("cuda")]


def maybe_skip_quantization(
    quantization: Optional[str],
    *,
    dims: Optional[Iterable[int] | int] = None,
    device: Optional[torch.device | str] = None,
) -> None:

    # Don't skip if there is no quantization
    if quantization is None:
        return

    # Check if quantization scheme is supported
    if quantization == "fp8" and not fp8_available:
        print('skip:', reason_for_no_fp8)
        pytest.skip(reason_for_no_fp8)
    if quantization == "mxfp8" and not mxfp8_available:
        print('skip:', reason_for_no_mxfp8)
        pytest.skip(reason_for_no_mxfp8)

    if dims is not None:
        if not isinstance(dims, Iterable):
            dims = (dims,)
        if quantization == "fp8":
            if math.prod(dims[:-1]) % 16 != 0 or dims[-1] % 16 != 0:
                print('skip: FP8 GEMMs require dims that are divisible by 16')
                pytest.skip("FP8 GEMMs require dims that are divisible by 16")
        elif quantization == "mxfp8":
            if math.prod(dims[:-1]) % 32 != 0 or dims[-1] % 32 != 0:
                print('skip: MXFP8 GEMMs require dims that are divisible by 32')
                pytest.skip("MXFP8 GEMMs require dims that are divisible by 32")

    # Check if device is supported
    if device is not None and torch.device(device).type != "cuda":
        print('skip: Quantization is only supported on CUDA devices')
        pytest.skip("Quantization is only supported on CUDA devices")


@torch.no_grad()
def make_reference_and_test_tensors(
    shape: int | Iterable[int],
    ref_dtype: torch.dtype = torch.float64,
    ref_device: torch.device = "cpu",
    test_dtype: torch.dtype = torch.float32,
    test_device: torch.device = "cuda",
    test_is_fp8: bool = False,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct tensors with the same values

    The reference tensor is intended for use in plain PyTorch
    operations in high precision. The test tensor is intended for use
    in Transformer Engine operations.

    """
    ref = torch.rand(shape, dtype=ref_dtype, device=ref_device)
    test = ref.to(device=test_device, dtype=test_dtype)
    if test_is_fp8:
        if Float8Quantizer:
            quantizer = Float8Quantizer(
                scale=torch.ones(1, dtype=torch.float32, device=test_device).squeeze(),
                amax=torch.zeros(1, dtype=torch.float32, device=test_device),
                fp8_dtype=tex.DType.kFloat8E4M3,
            )
            test = quantizer(test)
        else:
            test = Float8Tensor.to_float8(test)
            test._transpose = test._data.reshape(-1, test.size(-1)).transpose(0, 1)
            test._transpose = test._transpose.contiguous()
            test._transpose_invalid = False
    elif test.data_ptr() == ref.data_ptr():
        test = test.clone()
    ref.copy_(test)
    ref.requires_grad_(requires_grad)
    test.requires_grad_(requires_grad)
    return ref, test


def make_recipe(name: Optional[str] = None) -> Optional[Recipe]:
    """Make recipe for quantization scheme"""
    if name is None:
        return None
    if name == "fp8":
        return transformer_engine.common.recipe.DelayedScaling(
            fp8_format=transformer_engine.common.recipe.Format.E4M3,
        )
    if name == "mxfp8":
        return transformer_engine.common.recipe.MXFP8BlockScaling(
            fp8_format=transformer_engine.common.recipe.Format.E4M3,
        )
    raise ValueError(f"Unsupported quantization scheme ({name})")


def dtype_tols(dtype: torch.dtype | tex.DType) -> dict[str, float]:
    """Estimated numerical error for a datatype

    Based on tolerances for torch.testing.assert_close.

    """

    # Transformer Engine dtypes
    if isinstance(dtype, tex.DType):
        if dtype == tex.DType.kFloat8E4M3:
            return dict(rtol=0.125, atol=0.0675)  # epsilon = 0.0625
        if dtype == tex.DType.kFloat8E5M2:
            return dict(rtol=0.25, atol=0.125)  # epsilon = 0.152
        dtype = {
            tex.DType.kByte: torch.uint8,
            tex.DType.kInt32: torch.int32,
            tex.DType.kFloat32: torch.float32,
            tex.DType.kFloat16: torch.half,
            tex.DType.kBFloat16: torch.bfloat16,
        }[dtype]

    # PyTorch dtypes
    if dtype == torch.float16:
        return dict(rtol=1e-3, atol=1e-5)
    if dtype == torch.bfloat16:
        return dict(rtol=5e-3, atol=1e-5)
    if dtype == torch.float32:
        return dict(rtol=1.3e-6, atol=1e-6)
    if dtype == torch.float64:
        return dict(rtol=1e-7, atol=1e-7)
    raise ValueError(f"Unsupported dtype ({dtype})")


class TestBasicOps:
    """Tests for individual operations"""

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @pytest.mark.parametrize("weight_shape", ((19,), (64,), (5120,), (7168,), (8192,)))
    @pytest.mark.parametrize("in_shape", ((-1,), (6, 16, -1), (4096, -1)))
    @pytest.mark.parametrize("dtype", _dtypes)
    @pytest.mark.parametrize("zero_centered_gamma", (False, True))
    @pytest.mark.parametrize("quantization", (None, "fp8"))
    # @pytest.mark.parametrize("quantization", (None, "fp8", "mxfp8"))
    def test_rmsnorm(
        self,
        *,
        weight_shape: Iterable[int],
        in_shape: Iterable[int],
        dtype: torch.dtype,
        device: torch.device = "cuda",
        eps: float = 0.3,
        zero_centered_gamma: bool,
        quantization: Optional[str],
    ) -> None:
        """Layer norm"""

        # Make input and weight shapes consistent
        in_shape = list(in_shape)[:-1] + list(weight_shape)

        # Skip invalid configurations
        maybe_skip_quantization(quantization, dims=in_shape, device=device)

        # Random data
        x_ref, x_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
        )
        w_ref, w_test = make_reference_and_test_tensors(
            weight_shape,
            test_dtype=dtype,
            test_device=device,
        )
        dy_ref, dy_test = make_reference_and_test_tensors(
            in_shape,
            test_dtype=dtype,
            test_device=device,
            requires_grad=False,
        )

        # Plain PyTorch implementation
        inner_dims = tuple(range(len(in_shape) - len(weight_shape), len(in_shape)))
        var_ref = x_ref.square().sum(dim=inner_dims, keepdim=True) / math.prod(weight_shape)
        if zero_centered_gamma:
            y_ref = x_ref / torch.sqrt(eps + var_ref) * (1 + w_ref)
        else:
            y_ref = x_ref / torch.sqrt(eps + var_ref) * w_ref
        y_ref.backward(dy_ref)

        # Implementation with fusible operation
        op = te_ops.RMSNorm(
            weight_shape,
            eps=eps,
            device=device,
            dtype=dtype,
            zero_centered_gamma=zero_centered_gamma,
        )
        with torch.no_grad():
            op.weight.copy_(w_test)
            del w_test
        quantized_compute = quantization is not None
        recipe = make_recipe(quantization)
        forward = te_ops.Sequential(
            op,
            te_ops.Quantize(forward=quantized_compute, backward=False),
        )
        with te.fp8_autocast(enabled=quantized_compute, fp8_recipe=recipe):
            y_test = forward(x_test)
        y_test.backward(dy_test)

        # Expected numerical error
        tols = dtype_tols(dtype)
        if quantized_compute:
            tols = dtype_tols(tex.DType.kFloat8E4M3)

        # Check results
        y_test = y_test.to(dtype=torch.float64, device="cpu")
        dx_test = x_test.grad.to(dtype=torch.float64, device="cpu")
        dw_test = op.weight.grad.to(dtype=torch.float64, device="cpu")
        print(in_shape, dtype, device, zero_centered_gamma, quantization)
        torch.testing.assert_close(y_test, y_ref, **tols)
        torch.testing.assert_close(dx_test, x_ref.grad, **tols)
        torch.testing.assert_close(dw_test, w_ref.grad, **tols)