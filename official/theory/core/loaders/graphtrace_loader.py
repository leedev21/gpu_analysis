import os
from theory.core.loaders.bridge import LoaderBase, register_loader
from theory.core.utils import collection_path, read_file, Parser
from theory.core import Op
from theory.core.common.op import LaunchOp
from theory.core.utils import Parser


@register_loader('graphtrace')
class Loader(LoaderBase):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.total = 0
        self.need_check = {}

    def load(self, backend, cfg):
        try:
            data = read_file('', cfg['path'])
        except:
            print('load failed')
            return
        self.total = 0
        for fill_line in data:
            line = fill_line.strip()
            if line.startswith('-'):
                n_tab = fill_line.index('-')
                if line[2:].startswith('param_name:'):
                    pass
                elif line[2:].startswith('inputs:'):
                    pass
                elif line[2:].startswith('name:'):
                    module_name = line.split('::')[1]
                    if not 'CustomOpDef' in module_name:
                        print(' '*n_tab, 'AA ', module_name)
                else:
                    print(' '*n_tab, line[2:-1])
            elif line.startswith('inputs:'):
                pass
            elif line.startswith('file:'):
                pass
            else:
                items = line.split('%')
                op_info = '%'.join(items[2:]).strip()
                op_name = op_info.split('(')[0]
                if op_name.startswith('nn.functional'):
                    continue
                traced_op = LaunchOp(op_info.replace('%', ''), Parser)
                optrace_obj = Op(traced_op, backend='trace', debug={})
                if optrace_obj[0] == 'framework' or optrace_obj[1] == 'skip':
                    pass
                elif optrace_obj[0]:
                    if optrace_obj[1] is None and optrace_obj[3] not in ['Float', None]:
                        print('\t-->', op_info)
                        print('\t\t', optrace_obj)
                else:
                    if op_name not in self.need_check:
                        self.need_check[op_name] = {'case': [op_info], 'n': 0}
                    self.need_check[op_name]['n'] += 1
        print('='*100)
        for op_name in self.need_check:
            print(op_name, self.need_check[op_name]['n'], self.need_check[op_name]['case'])
