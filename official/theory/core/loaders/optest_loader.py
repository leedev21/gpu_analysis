import os
import json
import torch
from theory.core.loaders.bridge import LoaderBase, register_loader


@register_loader('optest')
class Loader(LoaderBase):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.total = 0

    def load(self, backend, cfg):
        self.total = 0
        with open(os.path.join(cfg['path'], 'op_test.json')) as f:
            test_case = json.load(f)
        ops = torch.load(os.path.join(test_case['data_path'], test_case['model_trace']), map_location=torch.device('cpu'))
        test_res, test_detail = [], []
        for i, op_name in enumerate(test_case['ops']):
            if 'filter' in cfg and cfg['filter'] and op_name not in cfg['filter']:
                continue
            if op_name not in self.objs:
                self.objs[op_name] = []
            for k, op_args in enumerate(test_case['ops'][op_name]):
                for j, op_index in enumerate(test_case['ops'][op_name][op_args]):
                    if op_index < len(ops[op_name]):
                        # print(op_name, op_index, ops[op_name][op_index])
                        self.objs[op_name].append((op_index, ops[op_name][op_index]))
                        self.total += 1
                    else:
                        print(op_index, len(ops[op_name]))