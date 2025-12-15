import os
from omegaconf import OmegaConf, DictConfig
from theory.core.loaders.bridge import LoaderBase, register_loader
from theory.core.utils import collection_path
from theory.core.utils import read_file, read_json
import csv


def read_csv(file_path, typ='op'):
    if typ == 'op':
        rows = []
        with open(file_path, "r") as csvfile:
            read_res = csv.reader(csvfile, delimiter=",")
            OP_INDEX = 0
            for i, items in enumerate(read_res):
                if i == 0:
                    for ii, key in enumerate(items):
                        if key == 'Range':
                            OP_INDEX = ii
                            break
                else:
                    rows.append(items[OP_INDEX])
        return rows
    if typ == 'dict':
        rows = {}
        with open(file_path, "r") as csvfile:
            read_res = csv.reader(csvfile, delimiter=",")
            for i, items in enumerate(read_res):
                rows[items[0]] = items[-1]
        return rows

def read_csv_all(file_path):
    rows = []
    with open(file_path, "r") as csvfile:
        read_res = csv.reader(csvfile, delimiter=",")
        for i, items in enumerate(read_res):
            # print(i, items)
            rows.append(items)
    return rows


@register_loader('torch_profile')
class Loader(LoaderBase):

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.base_config_path = os.path.join(collection_path(), 'config/params/models')
        self.graph_objs = {}
        self.db = None

    @staticmethod
    def get_op(ops):
        for i, _dict in enumerate(ops):
            if 'cat' not in _dict or _dict['cat'] not in ['kernel', 'gpu_memcpy']:
                if 'cat' not in _dict or _dict['cat'] not in ['user_annotation']:
                    if 'cat' not in _dict or _dict['cat'] not in ['cuda_runtime'] or _dict['name'] not in ['cudaLaunchKernel', 'cudaLaunchKernelExC', 'cudaMemcpyAsync']:
                        if 'cat' not in _dict or _dict['cat'] not in ['cuda_driver'] or _dict['name'] not in ['cuLaunchKernel']:
                            continue
                        yield None, _dict['ts'], _dict['dur'], None, None, _dict['pid'], _dict['name'], _dict['args']['External id']
                        continue
                    yield None, _dict['ts'], _dict['dur'], None, None, _dict['pid'], _dict['name'], _dict['args']['External id']
                    continue
                yield None, _dict['ts'], _dict['dur'], None, None, _dict['pid'], _dict['name'], _dict['args']['External id']
                continue
            yield None, _dict['ts'], _dict['dur'], _dict['args']['device'], _dict['args']['stream'], _dict['pid'], _dict['name'], _dict['args']['External id']

    def load(self, backend, cfg):
        print('load:', cfg)
        if cfg.d:
            data = read_json('', cfg.d)
            ops = data['traceEvents']
        else:
            print('Err no cfg.d:', cfg)
        benchmark_obj = backend('OpBenchmark', 'benchmark_name')
        self.graph_objs[cfg.d] = benchmark_obj.load_nvtx('', '', self.get_op(ops), cfg.model, only_graph=True)
        return self.graph_objs

    def get_all(self):
        return self.graph_objs

    def __getitem__(self, key):
        return self.graph_objs[key]
