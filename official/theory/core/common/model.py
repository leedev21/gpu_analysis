from theory.core.common.module import Module
from copy import deepcopy


class Model(object):
    def __init__(self, cfg, name='', config=None, graph=None, customer=None, create_by_conf=False):
        self.name = name
        self.cfg = cfg['module']
        self.config = deepcopy(config)
        self.precision = {}
        self.run = 'training' if config.training else 'inference'
        self.set_config(graph, cfg['config'])
        if self.config['num_image_tokens']:
            self.config['seq_length'] += self.config['num_image_tokens']

        self.decoding = {}
        self.percent = {}
        self.create_by_conf = create_by_conf
        self.customer_config = customer
        self.sub_module = self.create_model_by_config(customer) if self.create_by_conf else self.create_model_by_loading(customer)
        # self.print()

    def set_config(self, graph, config):
        if config.get('modules'):
            self.config.set('modules', config.get('modules'))
        elif config.get('exec'):
            self.config.set('modules', config['exec'].get('workload'))
        else:
            print('Err: failed to load workload', config)
            exit()
        self.config.set('graph', graph)
        self.config.set('target', config.get('target'))

    def create_model_by_config(self, args, stage='fwd'):
        module_list = []
        module_cfg = self.config.modules
        for module_name in module_cfg:
            if module_name == 'optim' and stage == 'bwd':
                continue
            this_module = self.get_this_module(module_name, module_cfg[module_name], args)
            module_list.append((module_name, Module(None, module_name, stage, this_module, self.config)))
        return module_list

    def get_item(self, key, item, args):
        if key and key in args:
            value = args[key]
        elif isinstance(item, int) or isinstance(item, dict):
            value = item
        elif item in args:
            value = args[item]
        else:
            value = self.config.shape(item)
        return value

    def get_this_module(self, module_name, this_module, args):
        if 'shape' not in this_module:
            this_module['shape'] = []
        if 'loop' in this_module:
            this_module['num_layers'] = self.get_item(None, this_module['loop'], args)
        if 'decoding' in this_module:
            self.decoding[module_name] = self.get_item(None, this_module['decoding'], args)
        if 'percent' in this_module:
            self.percent[module_name] = self.get_item(None, this_module['percent'], args)
        this_module['size'] = deepcopy(this_module['shape'])
        for i in range(len(this_module['shape'])):
            this_shape = self.get_item(None, this_module['shape'][i], args)
            this_module['size'][i] = this_shape
        # print('this module:', this_module)
        return this_module

    def create_model_by_loading(self, args, stage='fwd'):
        module_list = []
        return module_list

    def apply_feature(self):
        pass

    def __call__(self, backend, runtime):
        runtime.new(self.config, backend)
        runtime.config('iter', 0)
        runtime.add_target(self.config)
        for module_name, module in self.sub_module:
            if module_name == 'optim':
                runtime.config('stage', 'optim')
            res = module(backend, runtime)
            scale = self.percent[module_name]['base'] if self.percent and module_name in self.percent and 'base' in self.percent[module_name] else 1
            runtime.load(module_name, res, scale)
        if self.run == 'training':
            runtime.config('stage', 'bwd')
            sub_module = self.create_model_by_config(self.customer_config, 'bwd') if self.create_by_conf else self.create_model_by_loading(self.customer_config, 'bwd')
            for module_name, module in sub_module:
                res = module(backend, runtime)
                scale = self.percent[module_name]['base'] if self.percent and module_name in self.percent and 'base' in self.percent[module_name] else 1
                runtime.load(module_name, res, scale)
        runtime.config('tokens_in', self.config['seq_length'] * self.config['micro_batch_size'])
        if self.decoding:
            runtime.config('text_to_image', self.config['seq_length'] < 500) # tmp check!!!
            self.config['kv_cache'] = self.config['seq_length']
            self.config['seq_length'] = 1
            runtime.config('decoding', True)
            for module_name in self.decoding:
                tokens = self.decoding[module_name]
                runtime.cfg_obj.name += f"Out{self.config.size2str(tokens)}"
                if 'decoding' in self.customer_config:
                    name = self.config.update_test_id({'inference': self.customer_config['decoding']})
                    runtime.cfg_obj.name += f"_{name}"
                runtime.config('tokens', tokens)
                for i in range(tokens):
                    runtime.config('iter', i + 1)
                    self.config['kv_cache'] += 1
                    this_module = self.get_this_module(module_name, self.config.modules[module_name], self.customer_config)
                    module = Module(None, module, 'fwd', this_module, self.config)
                    res = module(backend, runtime, args={'decoding': True})
                    scale = self.percent[module_name]['decoding'] if self.percent and module_name in self.percent and 'decoding' in self.percent[module_name] else 1
                    runtime.load(module_name, res, scale, stage='decoding')
                runtime('module', module_name, res)
            runtime.config('tokens_out', tokens * self.config['micro_batch_size'])
            runtime.config('kv_dim', (self.config.shape('KVR') + self.config.shape('QKR')) if self.config.tags().get('mla') else 2 * self.config.shape('H') * self.config.shape('D'))
        else:
            runtime.config('tokens_out', 0)
        if module_name in self.percent and self.percent[module_name].get('L3_requirement'):
            runtime.config('L3_requirement', self.percent[module_name]['L3_requirement'])
            runtime.config('optim', self.percent[module_name]['optim'])

        runtime.exec(backend, self.customer_config if self.customer_config else 'model')

    def __map__(self):
        if not self.cfg:
            return

    def __iter__(self):
        if not self.cfg:
            return
        for struct in self.cfg:
            for stage in self.cfg[struct]:
                for layers in self.cfg[struct][stage]:
                    for layer_name in layers:
                        if isinstance(layers[layer_name], list):
                            for op_info in layers[layer_name]:
                                if isinstance(op_info, dict):
                                    yield struct, stage, layer_name, op_info['name'], op_info
                                else:
                                    yield struct, stage, layer_name, op_info, {}
                        elif isinstance(layers[layer_name], dict):
                            yield struct, stage, layer_name, layer_name, layers[layer_name]
                        else:
                            yield struct, stage, layer_name, layers[layer_name], {}

    def print(self):
        for struct, stage, layer_name, op_name, op_info in self:
            op_info = op_name + ': ' + str(op_info)
            print(struct, stage, layer_name, op_info)

    def save(self):
        pass