from theory.core.common.report import Report
from copy import deepcopy


class Stream():
    def __init__(self, obj_name, config) -> None:
        # device: 当不同device执行不同的runtime时，需单独执行
        # runtime; 执行n个stream，根据sync逻辑顺序执行各个stream上workload，创建执行流并输出结果
        # stream; 将workload放入stream中，一个stream内顺序执行，2个stream间需要定义sync，支持多micro bs
        # workload; 一个workload为一个block的执行，输入硬件配置，输出含op，module层级的执行信息，并存储
        # blocks; 计算图按block和module进行构建，block会单独创建workload，module仅作递归调用至op
        # cudaLaunchKernelExC; 发布workload
        # runtime重点支持pp，ep，decoding，pd分离，rl等机制
        self.device = 'default'
        self.work_load  = dict()
        self.stack = list()

    def __call__(self, cls, item, data):
        pass

    def new(self):
        # new work load
        pass

    def add(self):
        # new op/module
        pass

    def __iter__(self):
        return self

    def __next__(self):
        pass


class Runtime():
    def __init__(self, obj_name, config) -> None:
        # device: 当不同device执行不同的runtime时，需单独执行
        # runtime; 执行n个stream，根据sync逻辑顺序执行各个stream上workload，创建执行流并输出结果
        # stream; 将workload放入stream中，一个stream内顺序执行，2个stream间需要定义sync，支持多micro bs
        # workload; 一个workload为一个block的执行，输入硬件配置，输出含op，module层级的执行信息，并存储
        # blocks; 计算图按block和module进行构建，block会单独创建workload，module仅作递归调用至op
        # cudaLaunchKernelExC; 发布workload
        # runtime重点支持pp，ep，decoding，pd分离，rl等机制
        self.device = 'default'
        self.streams = list()

    def __call__(self, cls, item, data):
        pass

    def new(self):
        # new work load
        pass

    def add(self):
        # new op/module
        pass


class RunnerBase():
    def __init__(self, obj_name, config) -> None:
        self.reporter = Report(obj_name)
        self.cfg_obj = config
        self.run = 'training' if config.training else 'inference'
        self.summary = Summary()
        self.decoding = False
        self.conf = None
        self.blocks = []
        self.stage = 'fwd'
        self.stacks = []
        self.precision = {
            'params': config['main_params_dtype'],
            'optim': config['exp_avg_dtype'] if config['exp_avg_dtype'] == config['exp_avg_sq_dtype'] else None,
            'activation': config['main_grads_dtype'],
            'grad': config['main_grads_dtype'],
            'exp_avg': config['exp_avg_dtype'],
            'exp_avg_sq': config['exp_avg_dtype'],
        }
        self.state = dict()
        self.work_load = dict()
        self.fusion = dict()
        # experts: total experts; 存储分配
        # experts: activation experts; act experts per cards; work load;
        # stage：定义训练阶段
        # blocks: 在外面定义
        # 调度器, load profile负载
        # mla task

    def new(self, config, backend):
        self.run = 'training' if self.cfg_obj.training else 'inference'
        self.decoding = False
        self.stage = 'fwd'
        self.blocks.clear()
        self.stacks.clear()
        self.state.clear()
        self.work_load.clear()
        self.conf = None
        self.summary = Summary()
        self.state = config.tags('runner')
        self.cfg_obj = config
        self.stream = {}
        self.debug = False
        self.config('device', backend.name)
        self.cfg_obj.check(backend.name)

    def init_fusion(self, graph):
        for k, v in graph.items():
            if 'fusion' in v and v['fusion']:
                pattern = [a.split(';')[0] for a in v['base']['fwd']]
                # self.fusion[k] = {'name': k, 'pattern': pattern}
                self.fusion[pattern[0]] = {'name': k, 'pattern': pattern}
        for k, v in self.fusion.items():
            print('init_fusion:', k, v)

    def add_target(self, config):
        self.conf = config
        if config.target:
            self.reporter.add_target(config.target)

    def config(self, attr, value):
        if attr == 'decoding':
            self.decoding = value
            self.summary.set('seq_out', self.decoding, 'decoding')
        if attr == 'stage':
            self.stage = value
            return
        if attr in ['tokens_in', 'tokens_out', 'L3_requirement', 'kv_dim']:
            self.state[attr] = value
            return
        self.reporter.config(attr, value)

    def replay(self):
        for i in range(len(self.stacks)):
            self.reporter(*self.stacks[-i-1])

    def __call__(self, cls, item, data, stream='default'):
        if self.stream:
            self.stream.append(cls, item, data, stream)
        else:
            if self.stage == 'fwd':
                if cls == 'op':
                    if self.state.get('fusion_tag') and item == self.fusion[self.state['fusion_tag']]['pattern'][self.state['fusion_i']]:
                        self.stacks.append((cls, item, data, self.stage))
                        if self.state['fusion_i'] == len(self.fusion[self.state['fusion_tag']]['pattern']) - 1:
                            self.reporter(cls, self.fusion[self.state['fusion_tag']]['name'], data, self.stage)
                            self.stacks.clear()
                            self.state['fusion_tag'] = None
                            if self.debug:
                                print('end:', cls, item, data, self.stage)
                            # input()
                        else:
                            self.state['fusion_i'] += 1
                            if self.debug:
                                print('continue:', cls, item, data, self.stage)
                    elif item in self.fusion:
                        self.stacks.append((cls, item, data, self.stage))
                        self.state['fusion_tag'] = item
                        self.state['fusion_i'] = 1
                        if self.debug:
                            print('start:', cls, item, data, self.stage)
                    else:
                        if self.stacks:
                            for i in range(len(self.stacks)):
                                self.reporter(*self.stacks[i])
                            self.stacks.clear()
                            self.state['fusion_tag'] = None
                            if self.debug:
                                print('failed:', cls, item, data, self.stage)
                            # input()
                        self.reporter(cls, item, data, self.stage)
                else:
                    self.reporter(cls, item, data, self.stage)
            elif self.stage == 'bwd':
                self.stacks.append((cls, item, data, self.stage))
            else:
                self.reporter(cls, item, data, self.stage)

    def add_work_load(self, stage, value):
        if stage not in self.work_load:
            self.work_load[stage] = []
        self.work_load[stage].append(value)

    def load(self, module_name, res, scale, stage='base'):
        self.add_work_load(self.stage, res['latency'])
        if self.stage == 'bwd':
            self.add_work_load('weight_update', res['weight_update'])
        if stage == 'decoding':
            self.summary.add(res, scale, stage)
        else:
            self.summary.add(res, scale)
            self('module', module_name, res)

    def pipeline_parallel(self):
        if 'pipeline_model_parallel_size' in self.state and self.state['pipeline_model_parallel_size'] > 1:
            typ = self.cfg_obj['pipeline_type']
            PP = self.state['pipeline_model_parallel_size']
            M = self.state.get('gradient_accumulation_steps')
            F = self.work_load['fwd'][0]
            B = self.work_load['bwd'][0] if 'bwd' in self.work_load else 0
            FB = F + B
            W = self.work_load['weight_update'][0] if 'weight_update' in self.work_load else 0
            V = 0
            if 'virtual_pipeline_model_parallel_size' in self.state:
                typ = 'VirtualPipe'
                V = self.state['virtual_pipeline_model_parallel_size']
            return get_bubble(typ, PP, FB, F, B, W, V=V, M=M, cfg=self.cfg_obj)
        return {}

    def experts_parallel(self):
        '''
        "n_group": 8,
        "n_routed_experts": 256, # 非共享专家数
        "n_shared_experts": 1, # 共享专家书
        "num_experts_per_tok": 8, # 每个token激活抓夹数
        "num_hidden_layers": 61, #transformer层数
        "pretraining_tp": 1, # 预训练tp数
        topk2 * 8; choose top8
        '''
        if 'pipeline_model_parallel_size' in self.state and self.state['pipeline_model_parallel_size'] > 1:
            PP = self.state['pipeline_model_parallel_size']
            M = self.state['gradient_accumulation_steps']
            FB = self.work_load['fwd'][0] + self.work_load['bwd'][0]
            F = self.work_load['fwd'][0]
            B = self.work_load['bwd'][0]
            W = self.work_load['weight_update'][0]
            M = self.state['gradient_accumulation_steps']
            if 'virtual_pipeline_model_parallel_size' in self.state:
                typ = 'VirtualPipe'
                V = self.state['virtual_pipeline_model_parallel_size']
            return get_bubble(typ, PP, FB, F, B, W, V=V, M=M)
        return {}

    def _exec(self, device, backend, workload, streams, cfg):
        for stream in streams:
            pass

    def exec(self, backend, cfg):
        summary = self.summary
        if self.stage == 'bwd':
            self.replay()
        pp_ratio = self.pipeline_parallel()
        for key in ['2D', 'FA', 'comm', 'latency', 'total_latency']:
            for stage in ['default', 'decoding']:
                summary.compute('mul', key, 1 / 1000 / 1000, stage=stage)
        if pp_ratio:
            for stage in ['default', 'decoding']:
                for key in ['params', 'activation']:
                    print(key, pp_ratio[key])
                    summary.compute('mul', key, pp_ratio[key], stage=stage)
                # exit()
                for key in ['latency', 'total_latency', '2D_flops']:
                    summary.compute('mul', key, 1/8, stage=stage)
                # for key in ['latency', 'total_latency']:
                #     summary.compute('mul', key, 1/(1-pp_ratio['bubble_rate'][stage]), stage=stage)
            summary.set('bubble_rate', pp_ratio['bubble_rate']['default'], 'default')
            summary.set('bubble', pp_ratio['bubble']['default'] / 1000 / 1000, stage='default')
        if self.stage == 'bwd':
            summary.compute('mul', 'optim', backend.bpe(self.precision['optim']) * 3 / 1000 / 1000 / 1000, 'params', stage='default')
        if self.decoding:
            summary.data['default']['activation'] = self.state['kv_dim'] * (self.state['tokens_in'] + self.state['tokens_out']) * backend.bpe(self.precision['activation'])
            # print(self.state['kv_dim'], self.state['tokens_in'], self.state['tokens_out'], summary.data['default']['activation'])
        for key in ['params', 'activation']:
            for stage in ['default']:
                summary.compute('mul', key, backend.bpe(self.precision[key]) / 1000 / 1000 / 1000, stage=stage)
        summary.compute('add', 'total_mem', 'params', 'activation', stage='default')
        if self.stage == 'bwd':
            summary.compute('add', 'total_mem', 'optim', stage='default')

        if self.state.get('L3_requirement') and backend._storage['L3'] < self.state['L3_requirement']:
            zero1_latency = backend.compute_DD_in_Node('all_gather', [self.state['optim'], 1000000000], 'fp32') / 1000 / 1000
            latency_added = min(zero1_latency, summary.total()['default']['latency'])
            if latency_added != zero1_latency:
                summary.compute('mul', '2D_flops', 4/3, stage='default')
                summary.compute('mul', '2D_flops', latency_added, stage='decoding')
            else:
                for stage in ['default', 'decoding']:
                    summary.compute('add', 'comm', latency_added, stage=stage)
            for key in ['latency', 'total_latency']:
                for stage in ['default', 'decoding']:
                    summary.compute('add', key, latency_added, stage=stage)
            # summary.compute('add', 'comm', backend._storage['L3'] - self.state['L3_requirement'], stage='decoding')
        else:
            for stage in ['default', 'decoding']:
                summary.compute('add', 'comm', 0.001, stage=stage)

        summary.summary()
        summary.compute('div', 'tps', self.state['tokens_in'], 'total_latency', 'default')
        summary.compute('div', 'tps', self.state['tokens_out'], 'total_latency', 'decoding')
        summary.compute('div', 'tps', self.state['tokens_in'] + self.state['tokens_out'], 'total_latency', 'total')
        for stage in ['default', 'decoding', 'total']:
            summary.compute(backend.flops_utilization_cluster, 'MFU', self.conf, ('2D_flops', 'latency'), stage=stage)
        for key in ['comm', '2D', 'FA']:
            for stage in ['default', 'decoding', 'total']:
                summary.compute('div', key + '_percent', key, 'latency', stage=stage)
        summary.set('HW', backend.name)
        self.reporter('model', cfg, summary.total(), test_id=self.cfg_obj.name)

    def report(self):
        return self.reporter.report()


class ClusterRunner(RunnerBase):
    def __init__(self, cfg_obj, model, hw_object) -> None:
        super().__init__(cfg_obj, model, hw_object)
        self.obj_on_single_card = None
        self.obj_on_cluster = None
        self.obj_swap_by_tensor_parallel = None


class Summary():
    def __init__(self):
        self.data = {'default': {}, 'decoding': {}, 'total': {}}

    def add(self, values, scale, stage='default'):
        for k, v in values.items():
            if k in ['shape', 'op_type', 'num_layers']:
                continue
            if k in self.data[stage]:
                self.data[stage][k] += v
            else:
                self.data[stage][k] = v
        self.data[stage]['total_latency'] = self.data[stage]['latency'] / scale

    def compute(self, op, out, value, attr=None, stage='total'):
        if not self.data[stage]:
            return
        if not attr:
            attr = out
        if isinstance(value, str):
            value = self.data[stage][value]
        if op == 'mul':
            self.data[stage][out] = value * self.data[stage][attr]
        elif op == 'add':
            self.data[stage][out] = value + self.data[stage][attr]
        elif op == 'div':
            self.data[stage][out] = value / self.data[stage][attr]
        elif out == 'MFU':
            self.data[stage][out] = op(self.data[stage][attr[0]],
                                       # value['n_device'],
                                       1,
                                       value['precision'],
                                       self.data[stage][attr[1]])

    def set(self, key, value, stage='default'):
        self.data[stage][key] = value

    def summary(self):
        self.data['total'] = deepcopy(self.data['default'])
        if self.data['decoding']:
            for k, v in self.data['decoding'].items():
                if k in ['params', 'activation']:
                    continue
                if k not in self.data['total']:
                    self.data['total'][k] = v
                else:
                    self.data['total'][k] += v

    def total(self):
        return self.data


def get_bubble(typ, PP, FB, F=0, B=0, W=0, V=1, M=1, cfg=None):
    ratio = {'bubble': {'default': 0, 'decoding': 0},
             'bubble_rate': {'default': 0, 'decoding': 0},
             'params':1, 'activation': 1}
    if typ == 'chunked_prefill':
        ratio = {'bubble': {'default': 1/cfg['chunked_prefill']*F, 'decoding': 0.05*F},
                 'bubble_rate': {'default': 1/cfg['chunked_prefill'], 'decoding': 0.05},
                 'params':1/PP, 'activation': 1}
    if typ == '1F1B':
        ratio['bubble']['default'] = (PP-1) * (F+B)
        ratio.update({'params':1/PP, 'activation': 1})
    if typ == 'ZB1P':
        ratio['bubble']['default'] = (PP-1) * (F+B-W)
        ratio.update({'params':1/PP, 'activation': 1})
    if typ == 'ZeroPipe':
        ratio['bubble']['default'] = (PP/2-1) * (FB+B-3*W)
        ratio.update({'params':2/PP, 'activation': PP+1/PP})
    if typ == 'VirtualPipe':
        ratio['bubble']['default'] = (PP-1) * (F+B)
        ratio['bubble_rate']['default'] = (PP-1) / M / V
        ratio.update({'params':1/PP, 'activation': M//PP})
    if ratio['bubble_rate']['default'] == 0:
        ratio['bubble_rate']['default'] = ratio['bubble']['default'] / M * (F+B)
    return ratio