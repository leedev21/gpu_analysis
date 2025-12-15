from theory.core.collections import op
from theory.core.common.op import TimeObj
from copy import deepcopy


class Module(object):
    def __init__(self, ops, name='', stage='', module_info=None, module_cfg=None):
        self.name = name
        self.stage = stage
        self.ops = ops
        self.num_layers = 1
        self.launched_layers = 1
        self.module_info = module_info
        self.module_cfg = module_cfg
        self.type = None
        self.func_memory = None
        self.func_comm = None
        self.precision = {}
        self.debug = False
        self.fusion = module_info['fusion'] if 'fusion' in module_info else False
        self.sub_module = self.create_module_by_config()

    def get_this_module(self, this_module, shape, feature):
        if this_module.get('shape'):
            if not shape:
                shape = this_module['shape']
                self.shape[0] = this_module['shape']
            if isinstance(shape, str):
                shape = shape[1:-1].split(', ')
            # print('shape:', self.shape, this_module['shape'], shape)
            if isinstance(self.shape[1], list):
                this_module['size'] = self.get_shape_info(self.shape[0], self.shape[1], shape)
        else:
            if not shape:
                shape = self.shape[0]
            if isinstance(shape, str):
                shape = shape[1:-1].split(', ')
            # print('shape:', self.shape, shape)
            if isinstance(self.shape[1], list):
                this_module['size'] = self.get_shape_info(self.shape[0], self.shape[1], shape)
        if 'num_layers' in this_module and isinstance(this_module['num_layers'], str):
            try:
                this_module['num_layers'] = this_module['size'][this_module['shape'].index(this_module['num_layers'])]
            except:
                this_module['num_layers'] = self.module_cfg.shape(this_module['num_layers'])
        else:
            this_module['num_layers'] = 1
        return this_module, feature

    def get_value(self, item, input_item=None, input_size=None):
        def get_value(k, input_item, input_size):
            try:
                v = eval(k)
            except:
                if input_item and k in input_item:
                    v = input_size[input_item.index(k)]
                else:
                    v = self.module_cfg.shape(k)
            return v
        item = item.strip()
        scale = 1
        processed = False
        if '+' in item:
            items = item.split('+')
            value = 0
            for k in items:
                value += get_value(k.strip(), input_item, input_size)
            processed = True
        if '/' in item:
            items = item.split('/')
            if len(items) != 2:
                print('Not support Now:', items)
                exit()
            scale = get_value(items[1].strip(), input_item, input_size) if items[1] else 1
            item = items[0]
            processed = False
        if '*' in item:
            items = item.split('*')
            value = 1
            for k in items:
                value *= get_value(k.strip(), input_item, input_size)
            processed = True
        if not processed:
            value = get_value(item.strip(), input_item, input_size)
        try:
            value /= scale
        except:
            print('failed scale:', item, value, scale)
            exit()
        return value

    def get_shape_info(self, input_item, input_size, target_item):
        if self.debug:
            print(input_item, input_size, target_item)
        if isinstance(target_item, str):
            target_item = target_item[1:-1].split(',')
        target_size = []
        def get_value(k, input_item, input_size):
            try:
                v = eval(k)
            except:
                if k in input_item:
                    v = input_size[input_item.index(k)]
                else:
                    v = self.module_cfg.shape(k)
            return v
        for item in target_item:
            target_size.append(self.get_value(item, input_item, input_size))
        return target_size

    def create_module_by_config(self):
        module_list = []
        module_info = self.module_info
        self.num_layers = module_info['num_layers'] if module_info.get('num_layers') else 1
        self.launched_layers = self.get_value(module_info['launched_layers']) if module_info.get('launched_layers') else 1
        self.shape = [module_info['shape'], module_info['size']]
        module_with_no_feature = True
        for item in module_info:
            if item in self.module_cfg.tags() and self.module_cfg.tags()[item]:
                module_info = module_info[item]
                module_with_no_feature = False
                break
        if module_with_no_feature and 'base' in module_info:
            module_info = module_info['base']
        if self.debug:
            print('='*50, 'new', '='*50)
            print(self.name, self.module_info)
        stage = self.stage if self.stage in module_info else 'fwd'
        for item in module_info[stage]:
            feature = {}
            if ';' in item:
                items = item.split(';')
                item = items[0]
                i = 0
                if items[1].startswith('['):
                    shape = items[1]
                    i = 2
                else:
                    shape = []
                    i = 1
                for k in range(i, len(items)):
                    feature[items[k]] = True
            else:
                shape = []
            try:
                cls, module = item.split('::')
            except:
                print('Err split ::', item)
                exit()
            # print('next_module:', cls, module, shape)
            if cls == 'layer':
                this_module = self.get_this_module(deepcopy(self.module_cfg.graph[module]), shape, feature)
                # print('next:', cls, module, self.num_layers, self.shape, shape, this_module)
                module_list.append((item, (Module(None, module, self.stage, this_module[0], self.module_cfg), this_module[1])))
            else:
                this_module = self.get_this_module({}, shape, feature)
                # print('op:', cls, module, this_module)
                module_list.append((item, this_module))
        return module_list

    def __call__(self, backend, runtime, args={}):
        this_args = deepcopy(args)
        if self.fusion:
            this_args['fusion'] = True
        summary = {'2D': 0, 'FA': 0, '2D_flops': 0, 'comm': 0, 'comm_data': 0, 'weight_update': 0}
        for module_name, module in self.sub_module:
            module, feature = module
            if isinstance(module, dict):
                res = op.op_object(module_name, module['size'], backend, decoding=this_args.get('decoding'),
                                   precision=self.module_cfg.tags('runner').get('precision'), **feature)
                if 'c10d' in module_name:
                    res['latency'] = 0
                if runtime.stage in ['optim', 'bwd']:
                    del res['params']
                    del res['activation']
                elif res['activation']:
                    print('activation:', module_name, res['activation']/1000/1000/1000, 'G', module['size'], )
                if not this_args.get('fusion'):
                    fusion_det = runtime('op', module_name, res)
                # if res['op_type'] in ['2D', 'FA']:
                if res['op_type'] in ['FA']:
                    summary['FA'] += res['latency']
                elif res['op_type'] in ['2D']:
                    summary['2D'] += res['latency']
                elif res['op_type'] in ['Comm']:
                    summary['comm'] += res['latency']
            else:
                res = module(backend, runtime, this_args)
            if 'w_up' in feature:
                res['weight_update'] += res['latency']
            for k, v in res.items():
                if k in ['op_type', 'shape', 'B', 'M', 'K', 'N', 'num_layers', 'precision']:
                    continue
                if k not in summary:
                    summary[k] = 0
                if isinstance(v, dict):
                    for key in v:
                        for item in v[key]:
                            summary[k] += item
                else:
                    summary[k] += v
            if not isinstance(module, dict) and not this_args.get('fusion'):
                if not module.fusion:
                    runtime('module', module_name, res)
                else:
                    runtime('op', module_name, res)
        if self.num_layers > 1:
            for k in summary:
                if k in ['params', 'activation'] and self.module_info.get('loop'):
                    continue
                summary[k] *= self.num_layers
            if self.debug:
                print('********************************* check loop:', self.num_layers)
        elif self.launched_layers > 1:
            for k in summary:
                if k in ['params']:
                    summary[k] *= self.launched_layers
        summary['shape'] = self.shape
        summary['num_layers'] = self.num_layers
        return summary


class Block():
    def __init__(self, ops, name=None, stage=None, start= None, end= None, duration=None) -> None:
        self.ops = ops
        self.start = start
        self.end = end
        self.duration = None
        self.name = name
        self.stage = stage
        self.desp = set()
        self.add_duration(duration)

    def set(self, k, v):
        if k in ['duration']:
            self.add_duration(v)
        elif hasattr(self, k):
            setattr(self, k, v)
        if k in ['start', 'end']:
            self.add_duration(None)

    def add_duration(self, duration):
        if self.start is not None and self.end:
            if duration:
                if duration != self.end - self.start:
                    print('Err:', self.start, self.end, duration)
                    input('Pause')
                else:
                    self.duration = duration
            else:
                self.duration = self.end - self.start
        else:
            if duration:
                self.duration = duration
                if self.start is not None:
                    self.end = self.start + self.duration
                elif self.end:
                    self.start = self.end - self.duration
        if self.start is not None and self.end:
            self.desp.clear()
            for i in range(self.start, self.end):
                self.desp.add(self.ops[i].name)

    def print(self, name, ops=None, feature_dict={}):
        if not ops:
            ops = self.ops
        res = ['-'*20  + name + '-'*20]
        for i in range(self.start, self.end):
            line = [str(i), ops[i].name, ops[i].op_id, ':'.join(ops[i].stack)]
            if i in feature_dict:
                line.append(feature_dict[i].stage)
            res.append('->'.join(line))
        return res

    def print_tag(self, name=None, ops=None):
        res = str([self.start, self.end, self.end - self.start])
        return res

    def info(self, k):
        return self._info(k, typ='check')

    def _info(self, k, typ='std'):
        module_info = '{}_{} <{}, {}, {}>'.format(self.name, self.stage, self.start, self.end, self.duration)
        res = ['-'*20  + ' ' + str(k) + ':' + module_info  + ' ' + '-'*20]
        if typ == 'check':
            start = self.start - 10 if self.start >= 10 else 0
            end = self.end + 10 if self.end < len(self.ops) - 10 else len(self.ops) - 1
        else:
            start = self.start
            end = self.end
        jump = ''
        for i in range(start, end):
            if typ == 'check' and i > self.start + 10 and i < self.end - 10:
                jump += '.'
                continue
            tag = '-' if i < self.start else '+' if i >= self.end else ' '
            name = '  '*len(self.ops[i].stack) + self.ops[i].name
            if jump:
                res.append(jump)
                jump = ''
            line = '{} {} {:40} <{}> {}'.format(tag, i, name, self.ops[i].op_id,
                                              ':'.join(self.ops[i].stack))
            res.append(line)
        return res

    def desp_info(self):
        for i in range(self.start, self.end):
            yield self.ops[i]

    def match(self, ops_id, ops, back=True):
        matched = True
        module_index = self.end if back else self.start
        for i in range(100):
            index = -i if back else i
            op_object = ops[ops_id + index]
            module_object = ops[module_index + index]
            if module_object.name != op_object.name:
                matched = False
                break
        if i > 10:
            matched = True
        if not matched:
            return self._match(ops_id, ops, back)
        return matched, i, self.end - self.start - i

    def _match(self, ops_id, ops, back=True):
        block_zise = self.duration
        matched = {'pass': 0, 'fail': 0}
        last_match = 0
        for i in range(block_zise):
            index = -i if back else i
            op_object = ops[ops_id + index]
            if op_object.name in self.desp:
                matched['pass'] += 1
                last_match = index
            else:
                matched['fail'] += 1
        if matched['pass'] > block_zise * 0.9:
            matched = True
            print(ops_id, last_match, self.end - self.start - last_match)
            for i in range(ops_id + last_match - 1, ops_id + 1):
                print('\t', ops[i].name)
        else:
            matched = False
        return matched, last_match, self.end - self.start - last_match

    def leaky_match(self, desp):
        # print('leaky_match:', len(desp), len(self.desp))
        if desp == self.desp:
            return True
        return False


class RuntimeModule(Block, Module):
    def __init__(self, ops, block=None, name=None, stage=None, start= None, end= None) -> None:
        self.idx = 0
        self.inputs = []
        self.outputs = []
        self.theory_duration = 0
        self.desp = set()
        self.flops_utilization = 0.0
        self.use_count = 0
        self.memory = {}
        self.comm = {}
        if block:
            super().__init__(ops, block.name, block.stage, block.start, block.end, block.duration)
        else:
            super().__init__(ops, name, stage, start, end)

    def set(self, k, v):
        if hasattr(self, k):
            setattr(self, k, v)

    def print_module_info(self, device_id, index=0):
        module_info = self.module_info()
        duration = TimeObj(self.time('end') - self.time('start'), unit='ns')
        line = f"{device_id}:{module_info}-> {duration.to_str()}"
        return index, line

    def report_module_info(self):
        duration = TimeObj(self.time('end') - self.time('start'), unit='ns')
        info = '<{}, {}, {}> {}'.format(self.start, self.end, (self.end-self.start+1), duration.to_str())
        return self.name, self.stage,  info

    def module_info(self):
        return '{}_{} <{}, {}, {}>'.format(self.name, self.stage, self.start, self.end, self.duration)

    def block(self, *args, **kargs):
        return Block(self.ops, *args, **kargs)

    def new(self, *args, **kargs):
        return RuntimeModule(self.ops, *args, **kargs)

    def time(self, attr):
        id = getattr(self, attr)
        if attr == 'end':
            id -= 1
        return getattr(self.ops[id], attr)

    def module_match(self, block):
        if not self.leaky_match(block.desp):
            # print('leaky_match: failed')
            return False
        if self.name and block.name and self.name != block.name:
            # print('name_match: failed')
            return False
        if self.stage and block.stage and self.stage != block.stage:
            # print('stage_match: failed')
            return False
        return True


# class RuntimeModule(Module):
#     def __init__(self, ops, name='', stage='', start=None, end=None, duration=None) -> None:
#         super().__init__(ops, name, stage)
#         self.idx = 0
#         self.inputs = []
#         self.outputs = []
#         self.use_count = 0
#         self.start = start
#         self.end = end
#         self.duration = 0.0
#         self.theory_duration = 0
#         self.desp = set()
#         self.memory = {}
#         self.comm = {}
#         self.flops_utilization = 0.0


class RuntimeModuleSum(Module):
    def __init__(self):
        super().__init__()
        self.params = []
        self.params_count = 0
        self.use_count = 0
        self.flops_utilization = 0.0
