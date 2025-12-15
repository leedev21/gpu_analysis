import os
from theory.core.utils import collection_path, each_file, read_yaml_by_omegaconf, Parser
from theory.core.collections.op.bridge import register_manager


@register_manager('basic')
class OpManager(object):
    def __init__(self, config=None):
        self.objs = {}
        self.load(os.path.join(collection_path(), 'op/conf/basic'))

    def load(self, path, sep='\t'):
        for basic_file in each_file(path, '.yaml'):
            op_config = read_yaml_by_omegaconf(basic_file)
            for op_name in op_config:
                self.objs[op_name] = op_config[op_name]
                if 'type' in self.objs[op_name]:
                    self.objs[op_name]['type'] = self.objs[op_name]['type'].split(',')
                if 'schema' in self.objs[op_name]:
                    self.objs[op_name]['schema_dict'] = {}
                    for line in op_config[op_name]['schema']:
                        schema = line.split('->')
                        inputs = Parser('', schema[0])
                        outputs = Parser('', schema[1], out=True)
                        schema_for_match = ','.join([_in['value'] for _in in inputs.get()]) + '->' + ','.join([_in['value'] for _in in outputs.get()])
                        self.objs[op_name]['schema_dict'][schema_for_match] = {'only_shape': {'inputs': inputs.get('only_shape'),
                                                                                              'outputs': outputs.get('only_shape')}}

    def get_schemas(self, op_name):
        if op_name in self.objs and 'schema_dict' in self.objs[op_name]:
            return self.objs[op_name]['schema_dict']
        return self.objs['op::Base']['schema_dict']

    def __getitem__(self, key):
        return self.objs[key]

    def __setitem__(self, key, value):
        self.objs[key] = value

    def __iter__(self):
        for key in self.objs:
            yield key, self.objs[key]

    def __call__(self, op_name, shape, backend=None):
        pass

