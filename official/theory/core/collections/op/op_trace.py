import os
from theory.core.utils import collection_path, each_file, read_yaml_by_omegaconf, Parser
from theory.core.collections.op.bridge import register_manager


@register_manager('trace')
class OpManager(object):
    def __init__(self, config=None):
        self.objs = {}
        self.print_log = False
        self.load(os.path.join(collection_path(), 'op/conf/trace'))

    def load(self, path, sep='\t'):
        for basic_file in each_file(path, '.yaml'):
            op_config = read_yaml_by_omegaconf(basic_file)
            for op in op_config:
                self.objs[op] = op_config[op]

    def __call__(self, op_manager, op, backend=None, mapping_model='only_shape', debug={}):
        op_name = op.name
        if op_name not in self.objs:
            return None, None, None, None
        elif self.objs[op_name].get('type') == 'framework':
            return 'framework', None, None, None
        self.print_log = op_name in debug and 'trace' in debug[op_name]
        if mapping_model == 'only_shape':
            if self.print_log:
                print('check in:', op_name, op.inputs.get(), op.outputs.get())
            if self.objs[op_name].get('dtype'):
                attr, _i = self.objs[op_name]['dtype'].split('_')
                print(attr, _i, op_name, getattr(op, attr).get()[int(_i)])
                dtype = getattr(op, attr).get()[int(_i)]['value']['dtype']
            else:
                # print('out0:', op_name, op.outputs.get())
                if op.outputs.get()[0]['args_type'] == 'Tensor':
                    dtype = op.outputs.get()[0]['value']['dtype']
                else:
                    dtype = None
            op_type = self.get_op_type(op_name)
            schemas = op_manager['basic'].get_schemas(op_type)
            print(op_name, op_type, schemas)
            if schemas is None:
                return None, None, None, None
            print(op_name, op_type)
            mapping_model = 'only_shape' if op_type != 'op::DTE' else 'dte'
            op_info = op.get_by_schema(mapping_model, self.objs[op_name].get('mapping'))
            if self.print_log:
                print('op_info:', self.objs[op_name], op_info, dtype)
            schema, match_info = self.mapping(op, op_info, schemas, mapping_model)
            if self.print_log and schema is None:
                print('\tNo Schema:', op_info)
            if op_name == 'c10d::_allgather_base_':
                print(op_type, schema, match_info, dtype)
                exit()
            return op_type, schema, match_info, dtype
        else:
            print('Not support:', mapping_model)
        return None, None, None, None

    def get_op_type(self, op_name):
        if 'call' in self.objs[op_name]:
            return self.get_op_type(self.objs[op_name]['call'])
        else:
            return self.objs[op_name].get('basic', 'op::Base')

    def mapping(self, op, op_info, schemas, mapping_model):
        if mapping_model == 'dte':
            return self.mapping_dte(op, op_info)
        return self.mapping_by_shape_only(op_info, schemas, mapping_model)

    def mapping_by_shape_only(self, op_info, schemas, mapping_model):
        def check(schema):
            matched = {}
            for i, key in enumerate(['inputs', 'outputs']):
                if len(schema[key]) != len(op_info[i]):
                    if self.print_log:
                        print('length not match:', len(schema[key]), len(op_info[i]))
                    return False, {}
                for j, a in enumerate(schema[key]):
                    b = op_info[i][j]
                    if a not in matched:
                        matched[a] = b
                    elif matched[a] != b:
                        if self.print_log:
                            print('not matched:', a, matched[a], b, matched)
                        return False, {}
            return True, matched

        for schema_id in schemas:
            if self.print_log:
                print('check schema_id:', schema_id)
            is_matched, match_info = check(schemas[schema_id][mapping_model])
            if is_matched:
                if self.print_log:
                    print('matched:', schema_id, match_info)
                return schema_id, match_info
        return None, None

    def mapping_dte(self, op, op_info):
        # print('inputs:', op.inputs.get())
        # print('outputs:', op.outputs.get())
        # print(op_info)
        if op_info[0][0]['dtype'] == op_info[1][0]['dtype'] and op_info[0][0]['is_contiguous'] == op_info[1][0]['is_contiguous']:
            return 'skip', None
        _len = 5 - len(op_info[1][0]['size'])
        shape = '<N' + 'x'.join(['M', 'K', 'N', 'H'][_len:-1]) + '>'
        return 'dte', f"{shape}->{shape}"

