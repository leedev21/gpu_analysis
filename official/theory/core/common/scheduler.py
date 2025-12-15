from theory.core.common.model import Model
from theory.core.common.runner import RunnerBase
from copy import deepcopy

class Scheduler():
    def __init__(self, obj_name, cfg=None, cfg_obj=None, hw_loader=None) -> None:
        self.name = obj_name
        self.cfg = cfg
        self.obj = cfg_obj
        if cfg_obj:
            self.model_cfg = cfg_obj.model_cfg
            self.training_cfg = cfg_obj.training_cfg
            self.feature_cfg = cfg_obj.feature_cfg
            self.n_device = cfg_obj.hw_cfg['n_device']
            self.res = cfg_obj.res
        self.hw_loader = hw_loader
        self.hw_object = self.hw_loader.get(self.obj.hw_cfg['hw'])
        self.model = None
        self.run_list = []
        self.cfg_cache = []
        self.runtime = RunnerBase(obj_name, self.obj)
        self.config_upper_bound = True

    def create_model(self, modules, graph, create_by_conf):
        cfg = self.cfg['hw_cfg'] if self.cfg and self.cfg.get('hw_cfg') else None
        if cfg and 'hw' in cfg:
            self.obj.hw_cfg = cfg
            self.hw_object = self.hw_loader.get(self.obj.hw_cfg['hw'])
        self.model = Model(modules, self.name, self.obj, graph, cfg, create_by_conf)

    def check_config(self):
        if self.cfg and self.cfg.get('hw_cfg'):
            for k, v in self.cfg['hw_cfg'].items():
                if isinstance(v, (list, dict)):
                    return True
        return False

    def create_schedule(self, modules, graph, create_by_conf):
        if self.obj['fusion']:
            self.runtime.init_fusion(graph)
        if self.check_config():
            self.cfg_cache = [modules, graph, create_by_conf]
            self.run_list.append({})
            for k, v in self.cfg['hw_cfg'].items():
                len_run_list = len(self.run_list)
                if isinstance(v, (dict, list)):
                    for i, _item in enumerate(v):
                        if i > 0:
                            for _i in range(len_run_list):
                                self.run_list.append(deepcopy(self.run_list[_i]))
                        for _i in range(len_run_list):
                            if isinstance(v, list):
                                self.run_list[i * len_run_list + _i][k] = _item
                            else:
                                self.run_list[i * len_run_list + _i][k] = {
                                    'name': _item,
                                    'cfg': v[_item],
                                }
                else:
                    for _i in range(len_run_list):
                        self.run_list[_i][k] = v
            for i, line in enumerate(self.run_list):
                print('schedule:', i, line)
        else:
            self.create_model(modules, graph, create_by_conf)

    def __call__(self):
        if not self.run_list:
            if not self.config_upper_bound:
                self.hw_object.config('efficiency', 'upper_bound', False)
            self.model(self.hw_object, self.runtime)
        else:
            modules, graph, create_by_conf = self.cfg_cache
            for cfg in self.run_list:
                if 'hw' not in cfg:
                    this_hw_object = self.hw_object
                else:
                    if isinstance(cfg['hw'], str):
                        this_hw_object = self.hw_loader.get(cfg['hw'])
                    else:
                        this_hw_object = self.hw_loader.get(cfg['hw']['name'])
                        if cfg['hw']['cfg']:
                            for attr in cfg['hw']['cfg']:
                                print(attr, cfg['hw']['cfg'])
                                for key in cfg['hw']['cfg'][attr]:
                                    this_hw_object.config(attr, key, cfg['hw']['cfg'][attr][key])
                        cfg['hw'] = cfg['hw']['name']
                self.obj.update(cfg)
                if not self.config_upper_bound:
                    this_hw_object.config('efficiency', 'upper_bound', False)
                self.model = Model(modules, self.name, self.obj, graph, cfg, create_by_conf)
                self.model(this_hw_object, self.runtime)
        return self.runtime.report()