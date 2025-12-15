import os


class OpManagerBase():
    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self.objs = {}

    def add(self, cls, cls_name):
        # cls_name = cls.__class__
        # print('OpManager add:', cls_name, cls)
        self.objs[cls_name] = cls(None)

    def __call__(self, op, shape=None, backend=None, *args, **kwargs):
        if isinstance(op, str):
            return self.objs[backend](op, shape, backend, *args, **kwargs)
        else:
            return self.objs[backend](self, op, backend, *args, **kwargs)

    def __getitem__(self, key):
        return self.objs[key]

    def __setitem__(self, key, value):
        self.objs[key] = value

    def __iter__(self):
        for key in self.objs:
            yield key, self.objs[key]

op_manager = OpManagerBase()


def register_manager(cls_name):
    def wrapper(cls):
        op_manager.add(cls, cls_name)
        return cls
    return wrapper


def register_op(manager, cls_name):
    def wrapper(cls):
        op_manager[manager].add(cls, cls_name)
        return cls
    return wrapper 