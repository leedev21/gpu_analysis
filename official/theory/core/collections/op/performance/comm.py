from theory.core.common.op import Op as OpBase
from .base import register_op


@register_op('comm')
class Op(OpBase):
    def __init__(self, name):
        super().__init__(name)

    def get_mkn(self, shape):
        res = {'in': [], 'out': []}
        if isinstance(shape, list):
            if isinstance(shape[0], list):
                if len(shape) >= 2:
                    res['in'] =  shape[:-1]
                    res['out'] =  shape[-1]
                if len(shape) == 1:
                    res['in'] =  [shape[0]]
                    res['out'] =  shape[0]
            else:
                res['in'] =  [shape]
                res['out'] =  shape
        return res

    def __call__(self, shape, **kwargs):
        data = self.get_mkn(shape)
        res = {
            'op_type': 'Comm',
            'shape': self._int(shape),
            'io': self.get_io(data['in'] if 'reduce_scatter' not in self.name else [data['out']]),
            'params': self.get_weight(data['in']),
            'activation': self.get_activation(data['out']),
        }
        return res

    @staticmethod
    def get_io(shape):
        io = 1
        for a in shape[0]:
            io *= a
        return io

    @staticmethod
    def get_weight(shape):
        return 0

    @staticmethod
    def get_activation(shape):
        return 0