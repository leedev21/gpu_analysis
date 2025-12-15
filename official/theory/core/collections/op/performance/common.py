from theory.core.common.op import Op as OpBase
from .base import register_op
from omegaconf.listconfig import ListConfig


@register_op('1D')
class Op(OpBase):
    def __init__(self, name, args=None):
        super().__init__(name)
        self.r = args
        self.act = name in ['aten::add', 'aten::mul', 'aten::sum', 'aten::fill_', 'aten::mean', 'aten::div_',
                            'aten::copy_', 'aten::contiguous', 'aten::cat', 'aten::add_',
                            'aten::topk', 'aten::sigmoid', 'aten::scatter_', 'aten::div', 'aten::sub',]

    def get_mkn(self, shape, with_output=True, **kwargs):
        res = {'m': -1, 'n': -1, 'b': -1, 'in': [], 'out': []}
        if isinstance(shape, list):
            if isinstance(shape[0], list):
                def get_size(v):
                    if isinstance(v, (list, ListConfig)):
                        if isinstance(v[0], (list, ListConfig)):
                            total = 0
                            for i in range(len(v)):
                                total += get_size(v[i])
                            return total
                        else:
                            total = 1
                            for i in range(len(v)):
                                total *= get_size(v[i])
                            return total
                    else:
                        return v
                if isinstance(shape[0][0], (list, ListConfig)):
                    for i in range(len(shape)):
                        shape[i] = [get_size(shape[i])]
                if len(shape) >= 2:
                    if with_output:
                        res['in'] = shape[:-1]
                        res['out'] = shape[-1]
                    else:
                        res['in'] = shape
                        res['out'] = shape[0]
                elif len(shape) == 1:
                    res['in'] =  [shape[0]]
                    res['out'] =  shape[0]
                if len(shape[0]) >= 2:
                    res['m'] = shape[0][-2]
                    res['n'] = shape[0][-1]
                    if len(shape[0]) == 3:
                        res['b'] = shape[0][-3]
            else:
                res['in'] =  [shape]
                res['out'] =  shape
        return res

    def __call__(self, shape, **kwargs):
        if 'cast_to_fp8' in self.name:
            data = self.get_mkn(shape[:-1], **kwargs)
        elif 'grouped_topk' in self.name:
            convert_shape = [shape[0], shape[1] + shape[2]]
            data = self.get_mkn(convert_shape, **kwargs)
        elif 'sgl_moe_align_block_size' in self.name:
            convert_shape = [shape[0]*shape[1] + shape[2]*shape[4]]
            data = self.get_mkn(shape[:-1], **kwargs)
        else:
            data = self.get_mkn(shape, **kwargs)
        if not self.r:
            return {'op_type': 'failed', 'flops': 0, 'io': 0, 'params': 0, 'activation': 0}
        res = {
            'op_type': '1D',
            'shape': self._int(shape),
            'M': self._int(data['m']),
            'N': self._int(data['n']),
            'flops': self.get_flops(data['in'], self.r['flops']),
            'io': self.get_io(data['in'], data['out'], self.r['io']),
            'params': self.get_weight(data['in'], **kwargs),
            'activation': self.get_activation(data['out'], self.act),
        }
        if data['b'] != -1:
            res['B'] = data['b']
        return res

    def get_flops(self, shape, r=1):
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
    def get_io(shape_in, shape_out, r=1):
        io = 0
        if len(shape_in) > 1:
            for _t in shape_in:
                tmp = 1
                for a in _t:
                    tmp *= a
                io += tmp
            tmp = 1
            for a in shape_out:
                tmp *= a
            io += tmp
        else:
            for _t in shape_in:
                tmp = r
                for a in _t:
                    tmp *= a
                io += tmp
        return io

    @staticmethod
    def get_weight(shape, **kwargs):
        if kwargs and kwargs.get('weight'):
            return shape[0][-1]
        return 0

    @staticmethod
    def get_activation(shape, r=False):
        if r:
            return 0
        activation = 1
        for a in shape:
            activation *= a
        return activation
