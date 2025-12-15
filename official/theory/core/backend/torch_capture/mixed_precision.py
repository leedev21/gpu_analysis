import torch

from torch._higher_order_ops import wrap, cond, out_dtype
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

from .utils import register_dispatch_mode

aten = torch.ops.aten

SLOW_OPS = {
    aten.mm,
    aten.addmm,
    aten.bmm,
    aten.baddbmm,
    aten.convolution,
    aten._convolution,
    aten.convolution_backward,
    aten._scaled_dot_product_efficient_attention,
    aten._scaled_dot_product_flash_attention,
    aten._scaled_dot_product_efficient_attention_backward,
    aten._scaled_dot_product_flash_attention_backward,
}

def cast_if_tensor(x):
    if isinstance(x, torch.Tensor):
        if x.dtype.is_floating_point and x.dtype != torch.float32:
            return x.to(torch.float32)
    return x

@register_dispatch_mode(wrap.wrap)
# @register_dispatch_mode(cond.cond)
@register_dispatch_mode(out_dtype.out_dtype)
class MixedPrecisionMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        func_packet = func._overloadpacket
        if func_packet in SLOW_OPS:
            args, kwargs = tree_map(cast_if_tensor, (args, kwargs))
        return func(*args, **kwargs)
