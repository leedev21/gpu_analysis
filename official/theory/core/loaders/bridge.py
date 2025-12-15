import os
from theory.core.utils import read_yaml_by_omegaconf


class LoaderBase(object):
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.objs = {}

    def load(self, cfg):
        return {}

    def eval(self):
        pass

    def train(self):
        pass

    def load_yaml(self, path, name=None, check=False):
        if name:
            file = os.path.join(path, name + '.yaml')
        else:
            file = path
        if os.path.exists(file):
            config_dict = read_yaml_by_omegaconf(file)
            if check:
                for k, data in config_dict.items():
                    print(k, data)
            return config_dict
        print('load yaml failed:', path, name)
        return None

    def add(self, model_name, key, attr, reset=True):
        self.objs[model_name].add(key, attr, reset)

    def register(self, model_name, args, reset=False):
        self.objs[model_name].register(args, reset)

    def get_all(self):
        return self.objs

    def __getitem__(self, key):
        return self.objs[key]

    def __setitem__(self, key, value):
        self.objs[key] = value

    def __iter__(self):
        for key in self.objs:
            yield key

    def __enter__(self):
        return self

    def close(self):
        self.objs.clear()

    def clear(self):
        self.objs.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class LoaderManager(object):
    def __init__(self, config=None):
        self.engine = {}

    def add(self, cls, cls_name):
        # cls_name = cls.__class__
        # print('Loader add:', cls_name, cls)
        self.engine[cls_name] = cls(None)

    def __repr__(self):
        return str(self.engine)

    def __contains__(self, e):
        return True if e in self.engine else False

    def get(self, e):
        return self.engine[e].get_all()

    def load(self, e, backend=None, cfg={}):
        self.engine[e].load(backend, cfg)
        return self.engine[e]


loader = LoaderManager()


def register_loader(cls_name):
    def wrapper(cls):
        loader.add(cls, cls_name)
        return cls
    return wrapper