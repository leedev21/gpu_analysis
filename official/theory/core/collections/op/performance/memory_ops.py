from theory.core.common.op import Op as OpBase
from .base import register_op

class MemoryOp(OpBase):
    def __init__(self, name, args=None):
        super().__init__(name)
        self.r = args if args is not None else {'flops': 1, 'io': 1}

    def get_mkn(self, shape):
        res = {'m': -1, 'n': -1, 'b': -1, 'in': [], 'out': []}
        if isinstance(shape, list):
            if isinstance(shape[0], list):
                if len(shape) >= 2:
                    res['in'] = shape[:-1]
                    res['out'] = shape[-1]
                if len(shape) == 1:
                    res['in'] = [shape[0]]
                    res['out'] = shape[0]
                if len(shape[0]) >= 2:
                    res['m'] = shape[0][-2]
                    res['n'] = shape[0][-1]
                    if len(shape[0]) == 3:
                        res['b'] = shape[0][-3]
            else:
                res['in'] = [shape]
                res['out'] = shape
        return res

    def __call__(self, shape, **kwargs):
        data = self.get_mkn(shape)
        if not self.r:
            return {'op_type': 'failed', 'flops': 0, 'io': 0, 'params': 0, 'activation': 0}
        res = {
            'op_type': '1D',
            'M': self._int(data['m']),
            'N': self._int(data['n']),
            'shape': self._int(shape),
            'flops': self.get_flops(data['in'], self.r['flops']),
            'io': self.get_io(data['in'], self.r['io']),
            'params': self.get_weight(data['in']),
            'activation': self.get_activation(data['out']),
        }
        if data['b'] != -1:
            res['B'] = data['b']
        return res

    def get_flops(self, shape, r=1):
        # 内存操作算子的FLOPS是基于元素数量的
        if isinstance(r, (float, int)):
            flops = r
            for a in shape[0]:
                flops *= a
        else:
            flops = {}
            for typ in r:
                flops[typ] = [self.get_flops(shape, r[typ])]
        return flops

    @staticmethod
    def get_io(shape, r=1):
        # 内存操作的IO量通常是输入张量的大小
        io = 0
        for _t in shape:
            tmp = r
            for a in _t:
                tmp *= a
            io += tmp
        return io

    @staticmethod
    def get_weight(shape):
        # 内存操作没有权重参数
        return 0

    @staticmethod
    def get_activation(shape):
        # 激活值是输出张量的大小
        return 0


@register_op('aten::copy_')
class Copy(MemoryOp):
    def __init__(self):
        super().__init__('aten::copy_', {
            'flops': 1,  # 每个元素一次操作
            'io': 2,     # 读一次，写一次
        })


@register_op('aten::slice')
class Slice(MemoryOp):
    def __init__(self):
        super().__init__('aten::slice', {
            'flops': 0,  # 切片操作基本没有计算
            'io': 1,     # 只读取必要的数据
        })


@register_op('aten::contiguous')
class Contiguous(MemoryOp):
    def __init__(self):
        super().__init__('aten::contiguous', {
            'flops': 0,  # 内存布局转换，计算量很小
            'io': 2,     # 读取源数据并写入新的内存区域
        }) 