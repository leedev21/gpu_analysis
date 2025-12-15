import os
from theory.core.loaders.bridge import LoaderBase, register_loader
from theory.core.utils import each_file, read_yaml_by_omegaconf


@register_loader('bossa_nova')
class Loader(LoaderBase):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.total = 0

    def load(self, backend, cfg):
        for path, file in each_file(cfg['path'], endwith='yaml', with_path=True):
            items = path.replace(cfg['path'], '').split('/')
            op_name = items[1].replace('optype_', '')
            op_shape = [int(i) for i in items[2].replace('shape_', '').split('-')]
            op_dtype = items[3].replace('dtype_', '')
            report = read_yaml_by_omegaconf(file)
            if 'total_latency' in report:
                total_latency = report['total_latency'] * 1000 * 1000
                core_util = report['core_util']
                l3_rw_bw_util = report['l3_rw_bw_util']
                print('|'.join([op_name, str(op_shape), op_dtype, str(total_latency), str(core_util), str(l3_rw_bw_util)]))
            elif 'config' in report:
                pass
        return self