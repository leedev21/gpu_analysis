import pytest
import torch
from launcher.moprobe.utils import acc_check, draw_manager
from launcher.runner.module.layers.quant import native_per_token_group_quant_fp8

# DTYPES = [torch.bfloat16, torch.float]
DTYPES = [torch.float]
# QUANT_DTYPES = [None, torch.float8_e4m3fn]
QUANT_DTYPES = [None]
NUM_TOKENS_HIDDEN_SIZES = [
    [1, 1024],
]


@pytest.mark.parametrize("op_name", ['randn'])
@pytest.mark.parametrize("device", ['cpu'])
@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("use_allclose", [True])
@pytest.mark.parametrize("use_detail_check", [True])
@pytest.mark.parametrize("noise", [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001])
def test_acc_check(
    op_name: str,
    device: str,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    use_allclose: bool,
    use_detail_check: bool,
    noise: float,
) -> None:
    draw_manager.set('draw_ulp', True)
    draw_manager.set('output_keys', {0: 'out'})
    target = torch.randn(num_tokens, hidden_size).to(dtype)
    # target = torch.exp2(torch.range(-126, 127))
    # target = torch.tensor([1], device='cuda')
    # target[0] = 1.0
    out = target + noise
    if quant_dtype:
        target_out, target_scales = native_per_token_group_quant_fp8(target, 128)
        torch_out, torch_scales = native_per_token_group_quant_fp8(out, 128)
        target_out = target_out# .to(torch.float)
        res = acc_check(f'{op_name}_noise{noise}', device, target_out, torch_out, use_allclose, use_detail_check)

        # res = acc_check(op_name, device, target_scales, torch_scales, use_allclose, use_detail_check)
    else:
        res = acc_check(f'{op_name}_noise{noise}', device, target, out, use_allclose, use_detail_check)

    draw_manager.draw({'hw': 'Test'})
