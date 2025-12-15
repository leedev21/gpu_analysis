from theory.core.common.op import Op as OpBase
from .base import register_op


@register_op('embedding')
class Op(OpBase):
    def __init__(self, name, args=None):
        super().__init__(name)
        self.r = args

    def get_mkn(self, shape):
        res = {'m': -1, 'k': -1, 'n': -1}
        if isinstance(shape, list):
            if isinstance(shape[0], list):
                res['m'] = shape[0][-1]
                res['e'] = shape[1][-2]
                res['n'] = shape[1][-1]
            else:
                res['m'] = shape[0]
                res['e'] = shape[1]
                res['n'] = shape[2]
        return res

    def __call__(self, shape, **kwargs):
        data = self.get_mkn(shape)
        res = {
            'op_type': 'Emb',
            'M': self._int(data['m']),
            'K': self._int(data['e']),
            'N': self._int(data['n']),
            'shape': self._int(shape),
            'io': self.get_io(data['m'], data['e'], data['n'], self.r['io']),
            'params': self.get_weight(data['m'], data['e'], data['n']),
            'activation': self.get_activation(data['m'], data['e'], data['n']),
        }
        return res

    @staticmethod
    def get_io(m, e, n, r):
        return m * n * r

    @staticmethod
    def get_weight(m, e, n):
        return e * n

    @staticmethod
    def get_activation(m, e, n):
        return m * n + 8 * m