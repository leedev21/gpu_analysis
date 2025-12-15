import os
import json
import torch
from theory.core.backend.bridge import BackendObject, register_object
from theory.core.common.op import TimeObj, RuntimeOp
from theory.core.common.module import RuntimeModule
from .kernel_op import KernelOp
from .time_line import TimeLine


class STATE():
    def __init__(self) -> None:
        self.start = 0
        self.end = 0
        self.total = 0
        self.args = None
        self.last_latency = 0
        self.max_latency = 0
        self.optim_duration = 0
        self.pre_post_gap = 0
        self.step_spliter = None
        self.iter_spliter = []
        self.module_obj = None
        self.modules = []

    def set(self, k, v):
        if hasattr(self, k):
            setattr(self, k, v)

    def step_group(self, step_no):
        return self.step_spliter.group[step_no]

    def add_modules(self, blocks, step):
        # print('add_modules:', step, len(blocks), len(self.modules))
        for iter, block in blocks:
            if not self.modules:
                self.modules.append(self.module_obj.new(block=block))
                continue
            module_found = False
            for b in self.modules:
                if b.module_match(block):
                    if step == -1:
                        step, iter = iter, -1
                    b.add(block, step, iter)
                    module_found = True
                    break
            if not module_found:
                self.modules.append(self.module_obj.new(block=block))


class G_Device():
    def __init__(self, device_id) -> None:
        self.ops = []
        self.device_id = device_id
        self.start = -1
        self.end = -1
        self.state = STATE()
        self.state.set('module_obj', RuntimeModule(self.ops))
        self.op_list = []
        # self.features = []
        # self.features_iter = []
        # self.tasks = [('split_step', self.ops, self.features, self.state, False),
        #               ('recognize_optim', self.ops, self.features, self.state, True),
        #               ('recognize_iter', self.ops, self.features_iter, self.state, True)
        #              ]
        # self.feature_creator = FeatureStack()
        # self.config = {}
        # self.training_info = {}
        # self.module_info = {}
        # self.analysis = {}
        # self.steps_dict = {}
        # self.gpu_dict = {}

    def add(self, op_object):
        if op_object.name == '':
            return
        elif op_object.name not in self.op_list:
            self.op_list.append(op_object.name)
        self.ops.append(op_object)
        self.state.total += 1

    def summary(self):
        self.start = 0
        self.end = self.state.total
        self.start = self.ops[self.start].start
        self.end = self.ops[self.end - 1].end
        self.state.add_modules([(0, self.state.module_obj.block(start=0, end=len(self.ops),
                                                          name='full', stage='fwd_bwd'))], -1)

    def schedule(self):
        self.summary()

    def len(self):
        return self.state.total


class Graph():
    def __init__(self) -> None:
        self.pids = set()
        self.graph_device = {}
        self.official_time = None
        # self.split_success = False
        self.device_id = -1
        # self.workers = {
        #     'split_step': 'split_step_by_max_nvtx_latency',
        #     'recognize_optim': 'split_step_by_avg_step_duration',
        #     'recognize_iter': 'recognize_iter_by_op_feature',
        # }
        # self.features = {
        #     'iter': ['aten::embedding', 'aten::embedding_dense_backward'],
        #     'step': ['rms_norm'],
        #     'recognize_iter': 'recognize_iter_by_op_feature',
        # }
        self.user_annotation_graph_ops = {}
        self.user_annotation_pid = 1166

    # def set(self, model):
    #     self.args = Args(model)

    def add(self, op_object):
        device = op_object.pid
        self.pids.add(device)
        if device not in self.graph_device:
            self.graph_device[device] = G_Device(device)
        self.graph_device[device].add(op_object)

    def create(self, ops, profiler_format):
        res = {}
        for line in ops:
            self.add(RuntimeOp(line, kernel_op=KernelOp))
        #device_id, torch_ops = max(self.graph_device.items(), key=lambda a: a[1].len())
        exclude_device_id = self.user_annotation_pid
        filtered_devices = {k: v for k, v in self.graph_device.items() if k != exclude_device_id}
        device_id, torch_ops = max(filtered_devices.items(), key=lambda a: a[1].len())
        self.pids = list(sorted(self.pids))
        # if all(device_id != self.pids[k] for k in self.args.ranks):
        #     print('Err: device ID not matched:', device_id, self.args.ranks, self.pids)
        #     exit(0)
        # else:
        #     print('Device ID:', device_id, self.args.ranks, self.pids)
        self.device_id = device_id
        # self.graph_device[device_id].create_workers(self.workers)
        # 执行分析调度
        self.graph_device[device_id].schedule()
        # if self.user_annotation_pid == -1:
        #     stats, res = self.graph_device[device_id].schedule(profiler_format, None)
        #     report = {
        #         'config': res['config'],                    # 配置信息
        #         'training_info': res['training_info'],      # 训练性能信息
        #         'module_info': res['module_info'],          # 模块统计信息
        #         'analysis': res['analysis']                  # 详细分析结果
        #     }
        # else:
        #     stats, res = self.graph_device[device_id].schedule(profiler_format, self.graph_device[self.user_annotation_pid])
        #     report = {
        #         'config': res['config'],                    # 配置信息
        #         'training_info': res['training_info'],      # 训练性能信息
        #         'module_info': res['module_info'],          # 模块统计信息
        #         'analysis': res['analysis'],
        #         'module': res['module']                   # 详细分析结果
        #     }
           
        return torch_ops.ops
        # if profiler_format == 'nsys':
        #     self.graph_device[device_id].create_workers(self.workers)
        # stats, res = self.graph_device[device_id].schedule(profiler_format)
        # return {'config': res['config'],
        #         'training_info': res['training_info'],
        #         'module_info': res['module_info'],
        #         'analysis': res['analysis']
        #         }, torch_ops.ops

    def available_device(self):
        k = self.device_id
        self.official_time = {
            'start': self.graph_device[k].start,
            'end': self.graph_device[k].end
        }
        return 0

    def device_modules(self):
        k = self.device_id
        print('state.modules:', len(self.graph_device[k].state.modules))
        return self.graph_device[k].state.modules

    def get_op_list(self, device_id):
        return self.graph_device[device_id].op_list


@register_object('profile')
class NsysAnalyzer(BackendObject):
    def __init__(self, obj_name, cfg) -> None:
        super().__init__(obj_name, cfg)
        self.graph = Graph()

    def load_nvtx(self, kernel_trace, nvtx_kern_sum, nvtx_proj_trace, only_graph=False):
        profiler_format = 'torch' if only_graph else 'nsys'

        # self.graph.set(model)
        torch_ops = self.graph.create(nvtx_proj_trace, profiler_format)
        device = self.graph.available_device()
        if only_graph:
            kernel_trace = torch_ops
        op_list = self.graph.get_op_list(self.graph.device_id)

        if all('aten' not in op for op in op_list):
            time_line = TimeLine(kernel_trace, None, profiler_format, device)
            op_info, check_ops = time_line.get(device)
        else:
            time_line = TimeLine(kernel_trace, self.graph.official_time, profiler_format, device)
            op_info, check_ops = time_line.merge(self.graph.device_modules(), nvtx_kern_sum, device)

        print('-'*50, 'op_list', '-'*50)
        for op in op_list:
            print(op)
        print('-'*50, 'has_kernel', '-'*50)
        for op in op_info:
            has_kernel = []
            for case in op_info[op]:
                if case['has_kernel'] not in has_kernel:
                    has_kernel.append(case['has_kernel'])
            print(op, has_kernel)
        print('-'*50, 'check_ops', '-'*50)
        for op in check_ops:
            print(op)

        return {}