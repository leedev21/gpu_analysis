from typing import Dict, Any
from collections import defaultdict

from torch.utils.flop_counter import FlopCounterMode
from torch._higher_order_ops import wrap, cond, out_dtype

from .utils import register_dispatch_mode

# FIXME: higher order ops do not work well with FlopCounterMode.
#        is this the proper way to walk through it?
@register_dispatch_mode(wrap.wrap)
# @register_dispatch_mode(cond.cond)
@register_dispatch_mode(out_dtype.out_dtype)
class OpCounter(FlopCounterMode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.op_counts: Dict[str, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        func_packet = func._overloadpacket
        for par in self.parents:
            self.op_counts[par][func_packet] += 1

        return super().__torch_dispatch__(func, types, args, kwargs)

    def get_counts(self):
        def _update_counts(name: str, result: dict, data: dict):
            for parent, counts in data.items():
                if parent not in result:
                    result[parent] = {}

                result[parent][name] = {}
                for k, v in counts.items():
                    result[parent][name][str(k)] = v

        result = {}
        _update_counts('flops', result, self.get_flop_counts())
        _update_counts('ops',   result, self.op_counts)
        return result
