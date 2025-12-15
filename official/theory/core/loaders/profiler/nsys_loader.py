import os
from omegaconf import OmegaConf, DictConfig
from .utils import Sqlite, nsys_stats_report, read_csv_all, read_csv
from theory.core.loaders.bridge import LoaderBase, register_loader


@register_loader('nsys')
class Loader(LoaderBase):

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.db = None

    @staticmethod
    def get_op(ops):
        for line in ops[1:]:
            # 0: desp
            # 1: start
            # 2: duration
            # 3: 
            # 4: nvtx_duration
            # 5: 
            # 6: pid
            # 7: 
            # 8: 
            # 9: 
            # 10: num_child
            # 11: op_id
            # 12: parent_id
            # 13: stack
            yield line[0],int(line[1]),int(line[2]),line[4],line[6],int(line[10]),line[11],line[12],line[13].split(':')

    def process_nsys_rep(self, args):
        if args['path'].endswith('nsys-rep'):
            sqlite_name = args['path'].replace('.nsys-rep', '')
            nvtx_method = 'aten_op_list'
            summary_method = 'nvtx_kern_sum'
            trace_method = 'nvtx_gpu_proj_trace'
            args['d'] = sqlite_name
            args['sqlite'] = sqlite_name + '.sqlite'
            args['nvtx'] = sqlite_name + f'_{nvtx_method}.csv'
            args['nvtx_kern'] = sqlite_name + f'_{summary_method}.csv'
            args['nvtx_proj'] = sqlite_name + f'_{trace_method}.csv'
            args['split_type'] = 'nvtx' # support aten, kernel, nvtx
            nsys_stats_report(sqlite_name, nvtx_method, args['nvtx'], args['path'])
            nsys_stats_report(sqlite_name, summary_method, args['nvtx_kern'], args['path'])
            nsys_stats_report(sqlite_name, trace_method, args['nvtx_proj'], args['path'])
        else:
            print('nsys not_support:', args['path'])
        return args

    def load(self, backend, cfg):
        cfg = self.process_nsys_rep(cfg)
        if cfg['nvtx']:
            nvtx_ops = read_csv(cfg['nvtx'])
        if cfg['split_type'] == 'aten':
            self.objs[cfg['sqlite']] = nvtx_ops
            return self.objs
        self.db = Sqlite(cfg['sqlite'])
        load_dict = self.db.time_line()
        time_line_nvtx = []
        nvtx_ops = []
        try:
            time_line_data = self.db.time_line_nvtx()
            all_nvtx = set()
            for line in time_line_data:
                items = line[2].split(' = ')
                nvtx = {}
                key = 'op'
                for item in items[:-1]:
                    value = item.rsplit(',', 1)
                    nvtx[key] = value[0]
                    key = value[1]
                nvtx[key] = items[-1]
                nvtx['start'] = line[0]
                nvtx['duration'] = line[1]
                nvtx['type'] = line[3]
                time_line_nvtx.append(nvtx)
                all_nvtx.add(nvtx['op'])
            for op in nvtx_ops:
                if op and op not in all_nvtx:
                    # print(op)
                    input('Pause')
        except:
            input('Pause')
        if cfg['nvtx_kern']:
            nvtx_kern_sum = read_csv(cfg['nvtx_kern'], typ='dict')
            print(len(nvtx_kern_sum))
            # for k in nvtx_kern_sum:
            #     print(k, nvtx_kern_sum[k])
        if cfg['nvtx_proj']:
            nvtx_proj_trace = read_csv_all(cfg['nvtx_proj'])
            print(len(nvtx_proj_trace))
            # for line in nvtx_proj_trace:
            #     print(line)
        if  cfg['split_type'] == 'nvtx':
            self.objs[cfg['d']] = backend.load_nvtx(load_dict, nvtx_kern_sum,
                                                 self.get_op(nvtx_proj_trace))
        else:
            print(f'split_type: {cfg['split_type']} not support, only support kernel/nvtx')
            exit(0)
        self.close()
        return self.objs

    def close(self):
        self.db.close()
