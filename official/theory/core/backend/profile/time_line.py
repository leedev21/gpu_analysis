import os
import bisect
from theory.core import utils
from theory.core.common.op import TimeObj, RuntimeOp
from theory.core.utils import collection_path


class TimeLine():
    def __init__(self, kernel_trace, official_time, profiler_format, device=0) -> None:
        self.timeline = []
        self.timeline_op = {}
        self.name = ''
        self.pid = set()
        self.stack = []
        self.op_info = {}
        self.op_duration = {}
        self.aten_op = set()
        self.profiler_format = profiler_format
        self.create_by_module(kernel_trace, official_time, device)
        self.base_config_path = os.path.join(collection_path(), '../utils')
        self.torch_support_op = [k.strip() for k in utils.read_file(self.base_config_path, 'support_by_torch.txt')]

    def create_by_module(self, kernel_trace, official_time, device):
        print('num of kernel_trace:', len(kernel_trace))
        for i, items in enumerate(kernel_trace):
            items_op = []
            if self.profiler_format == 'torch':
                sname = items.s_kernel
                _start = items.start
                _duration = items.duration
                deviceId= items.deviceId
                streamId= items.streamId
                PID= items.pid
                fnam= items.f_kernel
                
                items_op.append(sname)
                items_op.append(_start)
                items_op.append(_duration)
                items_op.append(deviceId)
                items_op.append(streamId)
                items_op.append(PID)
                items_op.append(fnam)
            elif self.profiler_format == 'nsys':
                sname, _start, _duration, deviceId, streamId, PID, fname = items
                items_op = items
            else:
                break

            if int(deviceId) != device:
                continue
            self.pid.add(PID)
            _start = int(_start)
            # print(_start, _start - official_time['start'], official_time['end'] - _start, official_time['end'] - official_time['start'])
            if official_time:
                if _start < official_time['start']:
                    continue
                if _start + int(_duration) > official_time['end']:
                    break
            self.timeline.append(_start)
            self.timeline_op[_start] = {'kernel': RuntimeOp(items_op)}

    def add_to_op_duration(self, op, has_kernel):
        if op.name not in self.op_duration:
            self.op_duration[op.name] = []
        if any(item['size'] == op.shape for item in self.op_duration[op.name]):
            return

        check_backend_name = ''
        if op.name.startswith(":"):
            check_backend_name = op.name[1:]

        self.op_duration[op.name].append({'size': op.shape,
                                          'duration': TimeObj(op.duration, unit='ns').to_str(),
                                          'has_kernel': has_kernel,
                                          'has_aten': op.num_child,
                                          'support_by_backend': check_backend_name in self.torch_support_op})

    def add_matched_op(self, op, kernel_names, has_kernel):
        op_name = op.name
        if op_name not in self.op_info:
            self.op_info[op_name] = [kernel_names]
            self.add_to_op_duration(op, has_kernel)
        else:
            matched = False
            for items in self.op_info[op_name]:
                if str(items) == str(kernel_names):
                    matched = True
                    break
            if not matched:
                self.op_info[op_name].append(kernel_names)
                self.add_to_op_duration(op, has_kernel)

    def get(self, device):
        duration = 0
        duration_list = []
        kernel_list = []
        for i in range(len(self.timeline)):
            tmp = self.timeline_op[self.timeline[i]]['kernel'].print(i, typ='kernel')
            duration += tmp
            duration_list.append(tmp)
            kernel_list.append(self.timeline_op[self.timeline[i]]['kernel'].s_kernel)
        return {}, kernel_list

    def merge(self, modules, nvtx_kern_map, device):
        print('num of timeline:', len(self.timeline))
        for k, module in enumerate(modules):
            # print(k, module.print_tag())
            # for line in module.info(k):
            #     print(line)
            print(module.print_module_info(device, k))
            i = bisect.bisect_right(self.timeline, module.time('start'))
            end = bisect.bisect_right(self.timeline, module.time('end'))
            for j, op_object in enumerate(module.desp_info()):
                if int(op_object.pid) not in self.pid:
                    continue
                if op_object.name in nvtx_kern_map:
                    kernel_op_full_name = nvtx_kern_map[op_object.name]
                    if op_object.start in self.timeline:
                        if self.timeline_op[op_object.start]['kernel'].f_kernel == kernel_op_full_name:
                            # print('                   ------------------->', 'matched')
                            pass
                duration = 0
                duration_list = []
                kernel_list = []
                while i < end and self.timeline[i] < op_object.start:
                    tmp = self.timeline_op[self.timeline[i]]['kernel'].print(i, typ='kernel')
                    duration += tmp
                    duration_list.append(tmp)
                    kernel_list.append(self.timeline_op[self.timeline[i]]['kernel'].s_kernel)
                    i += 1
                if kernel_list:
                    for n in range(len(self.stack)):
                        if any(self.stack[n].name.startswith(k) for k in ['nccl', 'record']) or \
                        any(k in self.stack[n].name for k in ['ncclGroupEnd', 'nvte_cublas_gemm']) and len(self.stack) != 1:
                            continue
                        if len(duration_list) > 1 and self.stack[n].duration == duration_list[0]:
                            self.add_matched_op(self.stack[n], kernel_list[0], has_kernel=True)
                            # duration -= duration_list[0]
                            duration_list.pop(0)
                            kernel_list.pop(0)
                            continue
                        if self.stack[n].duration >= duration:
                            if 'LinearWithGradAccumulationAndAsyncCommunicationBackward' == self.stack[n].name:
                                break
                            self.add_matched_op(self.stack[n], kernel_list, has_kernel=True)
                            break
                op_object.print(j)
                if op_object.name.startswith('aten'):
                    self.aten_op.add(op_object.name)
                if len(op_object.stack) == 0:
                    self.stack = [op_object]
                else:
                    while len(op_object.stack) < len(self.stack):
                        op = self.stack.pop(0)
                        if op.name.startswith('aten'):
                            self.add_to_op_duration(op, has_kernel=False)
                    self.stack.insert(0, op_object)
        check_ops = []
        for op in self.aten_op:
            if op not in self.op_duration:
                if op in self.torch_support_op:
                    self.op_duration[op] = [{'size': [],
                                            'duration': 0,
                                            'has_kernel': False,
                                            'has_aten': 0,
                                            'support_by_backend': True}]
                else:
                    check_ops.append(op)
        return self.op_duration, check_ops
