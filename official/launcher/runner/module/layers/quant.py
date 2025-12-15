import torch
from typing import Optional, Union

try:
    from vllm.platforms import current_platform
    FP8_DTYPE = current_platform.fp8_dtype()
except ImportError:
    pass

try:
    import deep_gemm
except ImportError:
    print('deep_gemm not found')
    deep_gemm = None

def as_float32_tensor(x: Union[float, torch.tensor]) -> torch.tensor:
    return torch.as_tensor(x, dtype=torch.float32, device='cuda')


def native_per_token_group_quant_fp8(x,
                                     group_size=128,
                                     eps=1e-10,
                                     dtype=torch.float8_e4m3fn):
    """Function to perform per-token-group quantization on an input tensor
    `x` using native torch."""
    assert x.shape[-1] % group_size == 0, ("the last dimension of `x` cannot "
                                           "be divisible by `group_size`")
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1,
                        keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax / fp8_max
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size, ))

    return x_q, x_s


# with padding version
def per_block_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert tensor to FP8 format with per-block scaling."""
    if x.dim() == 3:
        b, m, n = x.shape
        m = b*m
        x = x.view(-1, n)
    elif x.dim() == 2:
        m, n = x.shape
        b = -1
    # Reshape directly into blocks of size 128x128 where possible
    x_view = x.view(-1, 128, n // 128, 128) if m % 128 == 0 and n % 128 == 0 else x.view(1, m, 1, n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_q, x_s = x_scaled.view_as(x)[:m, :n].contiguous(), (
        x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    if b != -1:
        x_q = x_q.view(b, -1, n)
        x_s = x_s.view(b, -1, x_view.size(2))
    return x_q, x_s

# deep_gemm version
def per_block_cast_to_fp8_deep_gemm(
        x: torch.Tensor,
        block_size_n: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (deep_gemm.ceil_div(m, 128) * 128,
         deep_gemm.ceil_div(n, block_size_n) * block_size_n),
        dtype=x.dtype,
        device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, block_size_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


def ref_dynamic_per_token_quant(x: torch.tensor,
                                quant_dtype: torch.dtype,
                                scale_ub: Optional[torch.tensor] = None) \
        -> tuple[torch.tensor, torch.tensor]:

    assert quant_dtype in [torch.int8, FP8_DTYPE]
    if scale_ub is not None:
        assert quant_dtype == FP8_DTYPE

    qtype_traits = torch.iinfo(quant_dtype) if quant_dtype == torch.int8 \
            else torch.finfo(quant_dtype)
    qtype_traits_max = qtype_traits.max
    qtype_traits_min = qtype_traits.min
    qtype_max = as_float32_tensor(qtype_traits_max)
    s_1 = as_float32_tensor(1.0)
    s_512 = as_float32_tensor(512.0)

    # For fp8, in order to match the cuda kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.

    # Compute scales
    x_token_max, _ = x.abs().max(dim=-1)
    x_token_max = as_float32_tensor(x_token_max)
    if scale_ub is not None:
        x_token_max = x_token_max.clamp(max=scale_ub)
    scales = (x_token_max / qtype_max)[:, None]

    # Quant
    if quant_dtype == torch.int8:
        iscales = as_float32_tensor(s_1 / scales)
        torch_out = as_float32_tensor(x) * iscales
        torch_out = torch_out.round()
        torch_out = torch_out.clamp(qtype_traits_min,
                                    qtype_traits_max).to(quant_dtype)
    else:
        assert quant_dtype == FP8_DTYPE
        min_scaling_factor = s_1 / (qtype_max * s_512)
        scales = scales.clamp(min=min_scaling_factor)
        torch_out = as_float32_tensor(x) / scales
        torch_out = torch_out.clamp(qtype_traits_min,
                                    qtype_traits_max).to(quant_dtype)

    return torch_out, scales


# The int8 version is very similar. Incorporate the int8 version, like in
# ref_dynamic_per_token_quant, when we have a dynamic_per_tensor int8 quant
# kernel
def ref_dynamic_per_tensor_fp8_quant(x: torch.tensor) \
                    -> tuple[torch.tensor, torch.tensor]:

    fp8_traits = torch.finfo(FP8_DTYPE)
    fp8_traits_max = fp8_traits.max
    fp8_traits_min = fp8_traits.min
    fp8_max = as_float32_tensor(fp8_traits_max)
    one = as_float32_tensor(1.0)

    # For fp8, in order to match the cuda kernel output, we have to do exactly
    # the same operations as in the corresponding fp8 kernel to prevent
    # rounding errors.

    x_max = as_float32_tensor(x.abs().max())
    ref_scale = x_max / fp8_max
    ref_iscale = one / ref_scale
    ref_out = (as_float32_tensor(x) * ref_iscale).clamp(
        fp8_traits_min, fp8_traits_max).to(FP8_DTYPE)
    return ref_out, ref_scale.view((1, ))
