class Op(object):
    def __init__(self, name):
        self.name = name
        self.type = ''
        self._name = ''
        self.schema = ''
        self.distributions = ''
        self.support_by_backend = ''
        self.default_cls = None
        self.func_flops = None
        self.func_cycle = None
        self.precision = 'f16'

    @staticmethod
    def _int(tensor):
        if isinstance(tensor, list):
            return [int(k) if isinstance(k, float) and k.is_integer() else k for k in tensor]
        else:
            return int(tensor) if isinstance(tensor, float) and tensor.is_integer() else tensor

    def exec(self, res, backend, decoding=False, precision=None):
        precision = precision if precision else self.precision
        return backend.exec(self.name, res['op_type'], res, precision, decoding)

    def utilization(self, res, backend, duration):
        return backend.exec(self.name, res['op_type'], res, self.precision, duration=duration)


class LaunchOp(Op):
    def __init__(self, line='', parser=None, debug={}):
        self.size = {}
        self.shape = []
        self.inputs = []
        self.outputs = []
        self.debug = debug
        super().__init__('')
        if line:
            self.create_by_trace(line, parser)

    def print(self, index=-1):
        if index == -1:
            print(self.name, self.inputs.get(), self.outputs.get())
        else:
            print(index, self.name, self.inputs.get(), self.outputs.get())

    def create_by_trace(self, line, parser):
        data = line.split('%')
        op_info = data[-1].strip()
        op_info = op_info.split(' -> ')
        op_index = op_info[0].index('(')
        self.name = op_info[0][:op_index]
        self.inputs = parser(self.name, op_info[0][op_index+1:-1], debug=self.debug)
        self.outputs = parser(self.name, op_info[1], out=True, debug=self.debug)

    def get_by_schema(self, mapping_model, mapping=None):
        if mapping is not None and 'outputs' in mapping:
            outputs = self.outputs.get(mapping_model, mapping['outputs'])
        else:
            outputs = self.outputs.get(mapping_model)
        if mapping is None:
            inputs = self.inputs.get(mapping_model)
        elif 'inputs' in mapping:
            inputs = self.inputs.get(mapping_model, mapping['inputs'])
        else:
            inputs = self.inputs.get(mapping_model, mapping)
        return inputs, outputs


class RuntimeOp(LaunchOp):
    def __init__(self, line, kernel_op=None):
        super().__init__()
        self.idx = 0
        self.kernel = ''
        self.f_kernel = ''
        self.s_kernel = ''
        self.desp = ''
        self.use_count = 0
        self.cpu_duration = 0.0
        self.cuda_duration = 0.0
        self.theory_duration = 0.0
        self.has_kernel = ''
        self.has_child = ''
        self.gpu_kernel = ''
        self.cuda_kernel = ''
        self.flops_utilization = 0.0
        # perf info
        self.start = 0
        self.duration = 0
        self.end = 0
        # nvtx info
        self.deviceId = -1
        self.streamId = -1
        self.nvtx_duration = -1
        self.pid = 0
        self.num_child = 0
        self.op_id = -1
        self.parent_id = -1
        self.stack = []
        self.kernel_op = None
        if len(line) == 7:
            self.create_kernel(*line, kernel_op=kernel_op)
        elif len(line) == 8:
            self.create_kernel(*line, kernel_op=kernel_op)
        elif len(line) == 9:
            self.create_nvtx(*line)
        else:
            print('Err load op:', len(line), line)
            input('Pause')

    def create_nvtx(self, desp, start, duration, nvtx_duration, pid, num_child, op_id, parent_id, stack):
        nvtx_desp = {}
        self.desp = desp
        items = desp.split(' = ')
        key = 'op'
        if items[0].startswith(':'):
            items[0] = items[0][1:]
        for item in items[:-1]:
            value = item.rsplit(', ', 1)
            nvtx_desp[key] = value[0]
            key = value[1]
        if len(items) > 1:
            self.name = nvtx_desp.get('op')
            self.shape = nvtx_desp.get('sizes')
        else:
            self.name = desp
        self.start = start
        self.duration = duration
        self.nvtx_duration = nvtx_duration
        self.end = self.start + self.duration
        self.pid = pid
        self.deviceId = pid
        self.num_child = num_child
        self.op_id = op_id
        self.parent_id = parent_id
        self.stack = stack

    def create_kernel(self, sname, _start, _duration, deviceId, streamId, PID, fname, op_id=None, kernel_op=None):
        if not sname and kernel_op:
            kernel = kernel_op(fname)
            self.kernel_op = kernel
            self.s_kernel = kernel.get_kernel_name()
            self.name = kernel.name()[0]
            print(self.name, self.s_kernel, '\t\t\t', fname)
        else:
            self.s_kernel = sname
        self.f_kernel = fname
        self.start = int(_start)
        self.duration = int(_duration)
        self.end = self.start + self.duration
        self.deviceId = deviceId
        self.pid = PID
        self.streamId = streamId
        if op_id:
            self.op_id = op_id

    def print(self, k, typ='nvtx'):
        tab = ''
        if self.stack:
            tab = '  '*(len(self.stack) - 1)
        if typ == 'kernel':
            tab = f'              -->kernel: {self.deviceId}:{self.streamId} '
        line = f"{tab}%{k} = {self.name}{self.s_kernel} {self.shape} -> {TimeObj(self.duration, unit='ns').to_str()}"
        print('  ', line)
        return self.duration


class RuntimeOpSum(Op):
    def __init__(self):
        super().__init__()
        self.params = []
        self.params_count = 0
        self.use_count = 0
        self.has_kernel = ''
        self.has_child = ''
        self.gpu_kernel = ''
        self.cuda_kernel = ''
        self.flops_utilization = 0.0


class TimeObj():
    def __init__(self, _str, unit=None) -> None:
        val, scale, unit = self._time(_str, unit)
        self.val = val
        self.scale = scale
        self.unit = unit

    def to_str(self):
        scale = self.scale
        val = self.val
        unit = 'ns'
        if scale < 1 and val * scale > 1:
            unit = 'us'
            val *= scale
            scale = 1
        if scale == 1 and val * 0.001 > 1:
            unit = 'ms'
            val *= 0.001
            scale = 1000
        if scale == 1000 and val * 0.001 > 1:
            unit = 's'
            val *= 0.001
            scale = 1000000
        return f"{val:.3f}{unit}"

    def value(self):
        return self.val, self.scale, self.unit

    def get_value(self, unit='us'):
        return self.val * self.scale

    @staticmethod
    def _time(_str, unit=None):
        if _str is None or _str == '':
            return None
        if not unit:
            unit = _str[-2:]
            _str = _str[:-2]
        val = 0.0
        scale = 1
        def value(val):
            if isinstance(val, str):
                return eval(val.strip())
            else:
                return val
        if unit in ['μs', 'us']:
            val = value(_str)
            scale = 1
        elif unit == 'ns':
            val = value(_str)
            scale = 0.001
        elif unit == 'ms':
            val = value(_str)
            scale = 1000
        elif unit == 's':
            val = value(_str)
            scale = 1000000
        else:
            print('time:', _str)
            input('Err data')
        return val, scale, unit