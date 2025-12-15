class BackendObject(object):
    def __init__(self, obj_name, cfg) -> None:
        self.name = obj_name
        self.cfg = cfg


class BackendManager(object):
    def __init__(self, config=None):
        self.objs = {}
        self.engine = {}

    def add(self, cls, cls_name):
        # cls_name = cls.__class__
        # print('objects add:', cls_name, cls)
        self.objs[cls_name] = cls

    def add_func(self, cls, cls_name):
        # cls_name = cls.__class__
        # print('engine add:', cls_name, cls)
        self.engine[cls_name] = cls

    def __repr__(self):
        return str(self.engine)

    def __contains__(self, e):
        return True if e in self.engine else False

    def get(self, e, obj_name, cfg=None):
        return self.objs[e](obj_name, cfg)

    def get_func(self, e, *args, **kwargs):
        return self.engine[e](*args, **kwargs)


backend = BackendManager()


def register_object(cls_name):
    def wrapper(cls):
        backend.add(cls, cls_name)
        return cls
    return wrapper


def register_function(cls_name):
    def wrapper(cls):
        backend.add_func(cls, cls_name)
        return cls
    return wrapper


def create_benchmarkobj(type, obj_name='', cfg=None):
    return manager.get(type, obj_name, cfg)


def run(type, *args, **kwargs):
    return manager.get_func(type, *args, **kwargs)