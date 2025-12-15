import os
from omegaconf.omegaconf import OmegaConf
from theory.core.utils import collection_path


class OpManager(object):
    def __init__(self, config=None):
        self.objects = {}
        self.base_config_path = os.path.join(collection_path(), 'op/conf')
        self.read_aten_yaml(os.path.join(self.base_config_path, 'default.yaml'))

    def config_op_attr(self, op, attr, value):
        if isinstance(value, dict):
            if attr not in self.aten_ops[op]:
                self.aten_ops[op][attr] = {}
            elif not isinstance(self.aten_ops[op][attr], dict):
                self.aten_ops[op][attr]['1D'] = self.aten_ops[op][attr]
            for k in value:
                if k not in self.aten_ops[op][attr]:
                    self.aten_ops[op][attr][k] = 0
                self.aten_ops[op][attr][k] += value[k]
        else:
            if attr not in self.aten_ops[op]:
                self.aten_ops[op][attr] = 0
            if isinstance(self.aten_ops[op][attr], dict):
                if '1D' not in self.aten_ops[op][attr]:
                    self.aten_ops[op][attr]['1D'] = 0
                self.aten_ops[op][attr]['1D'] += value
            else:
                self.aten_ops[op][attr] += value

    def op_tree(self, op, father=None):
        if op not in self.aten_ops:
            return
        if 'flops' in self.aten_ops[op] or 'io' in self.aten_ops[op]:
            pass
        elif 'split' in self.aten_ops[op]:
            for child in self.aten_ops[op]['split']:
                not_found = True
                _child = f'aten::{child}'
                for k in ['aten', 'customer']:
                    _child = f'{k}::{child}'
                    if _child in self.aten_ops:
                        not_found = False
                        break
                if not_found:
                    print('Err child not found in op:', op, _child, self.aten_ops[op])
                    # exit()
                self.op_tree(_child, op)
        else:
            pass
            # print('Err child not found in op:', op, father, self.aten_ops[op])
            # exit()
        if father:
            # print('back:', op, father, self.aten_ops[op])
            if 'flops' not in self.aten_ops[op] and 'io' not in self.aten_ops[op]:
                return
            self.config_op_attr(father, 'flops', self.aten_ops[op]['flops'])
            self.config_op_attr(father, 'io', self.aten_ops[op]['io'])

    def read_aten_yaml(self, path, sep='\t'):
        self.aten_ops = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        for op in self.aten_ops:
            self.op_tree(op)
            # print(op, self.aten_ops[op])

    def add(self, cls, cls_name):
        # cls_name = cls.__class__
        print('op object add:', cls_name, cls)
        self.objects[cls_name] = cls

    def get_op_info(self, op_name):
        items = op_name.split('::')
        op_item = op_name
        if len(items) == 1:
            if not op_name.startswith('aten'):
                op_item = 'customer::' + op_name
        elif len(items) == 2:
            if not op_name.startswith('aten'):
                op_item = 'customer::' + items[1]
        return self.aten_ops.get(op_item)

    def __call__(self, op_name, shape, backend=None, duration=None, decoding=False, precision=None, **kwargs):
        print('perf:', op_name, shape)
        if op_name in self.objects:
            op = self.objects[op_name]()
        elif op_name.startswith('c10d'):
            op = self.objects['comm'](op_name)
        elif op_name.startswith('aten') and 'embedding' in op_name:
            op_info = self.get_op_info(op_name)
            op = self.objects['embedding'](op_name, args=op_info)
        elif op_name.startswith('aten') and 'conv' in op_name:
            op_info = self.get_op_info(op_name)
            op = self.objects['conv'](op_name, args=op_info)
        elif any(k in op_name for k in ['FlashAttn', 'flash_attn', 'attention']):
            op_info = self.get_op_info(op_name)
            bwd =  any(k in op_name for k in ['bwd', 'Backward'])
            op = self.objects['flash_attn'](op_name, args=op_info, bwd=bwd)
        else:
            op_info = self.get_op_info(op_name)
            if op_info and op_info.get('2D'):
                op = self.objects['2D'](op_name)
            else:
                op = self.objects['1D'](op_name, args=op_info)
        res = op(shape, decoding=decoding, **kwargs)
        if duration:
            res.update(op.utilization(res, backend, duration))
        elif backend:
            precision = precision if res['op_type'] in ['2D'] and op_name != 'aten::linear' else None   # add FA if enable fp8 flash attention
            if precision == 'fp8':
                res['precision'] = 'fp8'
            res['latency'] = op.exec(res, backend, decoding, precision)
        return res

    def __repr__(self):
        return str(self.engine)

    def __contains__(self, e):
        return True if e in self.engine else False

    def get(self, e, obj_name, cfg):
        return self.objects[e](obj_name, cfg)

    def get_func(self, e, *args, **kwargs):
        return self.engine[e](*args, **kwargs)

op_object = OpManager()

def register_op(cls_name):
    def wrapper(cls):
        op_object.add(cls, cls_name)
        return cls
    return wrapper
