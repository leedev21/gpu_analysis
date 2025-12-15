from theory.core.common.op import Op as OpBase
from .base import register_op


@register_op('conv')
class Op(OpBase):
    def __init__(self, name, args=None):
        super().__init__(name)

    def get_mkn(self, shape):
        res = {'in': [], 'out': []}
        if isinstance(shape, list):
            if len(shape) == 2:
                res['in'] =  shape[1]
                res['out'] =  shape[0]
        return res

    def __call__(self, shape, **kwargs):
        data = self.get_mkn(shape)
        res = {
            'op_type': '2D',
            'shape': self._int(shape),
            'flops': self.get_flops(data['in']),
            'io': self.get_io(data['in']),
            'params': self.get_weight(data['in']),
            'activation': self.get_activation(data['out']),
        }
        return res

    @staticmethod
    def get_flops(shape):
        flops = 1
        for a in shape:
            flops *= a
        return flops * 2

    @staticmethod
    def get_io(shape):
        io = 1
        for a in shape:
            io *= a
        return io

    @staticmethod
    def get_weight(shape):
        return 0

    @staticmethod
    def get_activation(shape):
        return 0