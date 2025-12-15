import os
from theory.core.loaders.bridge import LoaderBase, register_loader
from theory.core.common.op import LaunchOp
from theory.core.utils import collection_path, read_file, Parser


@register_loader('oplist')
class Loader(LoaderBase):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.total = 0

    def load(self, backend, cfg):
        data = read_file('', cfg['path'])
        if data:
            self.total = 0
            for line in data:
                line = line.strip()
                if line.startswith('%'):
                    traced_op = LaunchOp(line, Parser, cfg.get('debug', {}))
                    if traced_op.name not in self.objs:
                        self.objs[traced_op.name] = []
                    self.objs[traced_op.name].append(traced_op)
                    self.total += 1
        return self