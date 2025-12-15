from theory.core.common.op import Op as OpBase
from .base import register_op

@register_op('2D')
class Op(OpBase):
    def __init__(self, name):
        super().__init__(name)

    def get_mkn(self, shape):
        bs_is_1 = False
        res = {'m': -1, 'k': -1, 'n': -1, 'b': 1, 'bias': 0}
        if isinstance(shape, list):
            if isinstance(shape[0], list):
                if len(shape[0]) > 2:
                    for i in range(len(shape[0]) - 2):
                        res['b'] *= shape[0][i]
                t_list = {}
                if len(shape) > 2:
                    for i in range(3):
                        for j in range(1, 3):
                            if len(shape[i]) >= j:
                                if shape[i][-j] not in t_list:
                                    t_list[shape[i][-j]] = ''
                                t_list[shape[i][-j]] += str(i)
                    for a in t_list:
                        if t_list[a] == '02':
                            res['m'] = a
                        if t_list[a] == '01':
                            res['k'] = a
                        if t_list[a] == '12':
                            res['n'] = a
                        if t_list[a] == '0012' or t_list[a] == '0122':
                            if res['m'] == -1:
                                res['m'] = a
                            if res['k'] == -1:
                                res['k'] = a
                            if res['n'] == -1:
                                res['n'] = a
                    if len(shape) == 4:
                        res['bias'] = shape[-1]
                else:
                    res['m'] = shape[0][-2]
                    res['k'] = shape[0][-1]
                    res['n'] = shape[1][-1]
            else:
                if len(shape) == 5:
                    res['per_group'] = shape[-1]
                    shape = shape[:-1]
                if len(shape) >= 4:
                    res['b'] = shape[0]
                    if res['b'] == 1:
                        bs_is_1 = True
                res['m'] = shape[-3]
                res['k'] = shape[-2]
                res['n'] = shape[-1]
        return res, bs_is_1

    def run(self, shape, **kwargs):
        data, bs_is_1 = self.get_mkn(shape)
        flops_fwd = self.gemm_flops(data['m'], data['k'], data['n'], data['b'], data.get('per_group'))
        if bs_is_1:
            shape = shape[1:]
        res = {
            'op_type': '2D',
            'M': self._int(data['m']),
            'K': self._int(data['k']),
            'N': self._int(data['n']),
            'shape': self._int(shape),
            'flops': flops_fwd,
            '2D_flops': flops_fwd,
            'flops_fwd_bwd': 3 * flops_fwd,
            'io': self.get_io(data['m'], data['k'], data['n'], data['b']),
            'params': self.get_weight(data['m'], data['k'], data['n']),
            'activation': self.get_activation(data['m'], data['k'], data['n']),
        }
        if data['b'] != -1 and data['b'] != 1:
            res['B'] = data['b']
        return res

    def __call__(self, shape, **kwargs):
        return self.run(shape, **kwargs)

    @staticmethod
    def gemm_flops(m, k, n, b=1, per_group=-1):
        # 𝐴𝑚×𝑘 ×𝑋𝑘×𝑛 matrix multiplication requires 2𝑚 ×𝑘 ×𝑛 FLOPs
        if per_group and per_group > 1:
            return 2 * m * k * n * b
        else:
            return 2 * m * k * n * b

    @staticmethod
    def get_io(m, k, n, b=1):
        io = 0
        io += b * m * k
        io += b * k * n
        io += b * m * n
        return io

    @staticmethod
    def get_weight(m, k, n):
        return k * n

    @staticmethod
    def get_activation(m, k, n):
        return m * n


@register_op('aten::baddbmm')
class Baddbmm(Op):
    def __init__(self):
        super(Baddbmm, self).__init__('aten::baddbmm')

    def __call__(self, shape, **kwargs):
        return self.run(shape[1:], **kwargs)