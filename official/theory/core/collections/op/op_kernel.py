
class OpManager(object):
    def __init__(self, config=None, debug={}):
        self.objs = {}
        self.base_config_path = os.path.join(collection_path(), 'op/conf/basic')

    def load(self):
        pass

op_basic = OpManager(backend='exec')

class OpManager(object):
    def __init__(self, config=None):
        self.objects = {}
        self.base_config_path = os.path.join(collection_path(), 'op/conf/trace')

op_trace = OpManager(backend='trace')

class OpManager(object):
    def __init__(self, config=None):
        self.objects = {}
        self.base_config_path = os.path.join(collection_path(), 'op/conf/launch')

op_exec = OpManager(backend='exec')

class OpManager(object):
    def __init__(self, config=None):
        self.objects = {}
        self.base_config_path = os.path.join(collection_path(), 'op/conf/launch')



op_object = OpManager(backend='hardware')