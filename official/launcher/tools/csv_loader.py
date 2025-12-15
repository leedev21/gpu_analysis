from theory.core.utils import read_file, read_json, each_file
import torch
from copy import deepcopy
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig

def load_op_txt(path, max_cases_per_op=1):
    """
    加载算子测试数据，每个算子最多加载max_cases_per_op组数据
    
    Args:
        path: CSV文件路径
        max_cases_per_op: 每个算子最多加载的测试用例数量，默认30
    """
    op_dict = {}
    official_op_list = read_file('', path)
    for i, item in enumerate(official_op_list):
        for k in [',', '\t']:
            items = item.strip('\n').split(k)
            if len(items) >= 12:
                break
        op_name = items[4]
        if op_name == 'operator':
            continue
        op_info = eval(items[12].replace('，', ','))
        if op_name not in op_dict:
            op_dict[op_name] = {'inputs': [op_info['inputs']], 'aten_op_name': op_info['aten_op_name'].replace('.out', '')}
        else:
            # 限制每个算子最多获取max_cases_per_op组数据
            if len(op_dict[op_name]['inputs']) < max_cases_per_op:
                op_dict[op_name]['inputs'].append(op_info['inputs'])
            # 如果已经达到限制数量，跳过后续数据
    
    # 打印加载统计信息
    print(f"数据加载完成，每个算子最多加载 {max_cases_per_op} 组测试数据:")
    for op_name, op_info in op_dict.items():
        print(f"  算子 {op_name}: 加载了 {len(op_info['inputs'])} 组测试数据")
    
    return op_dict


class Loader(object):
    op_filter = []
    config_as_kwargs = False
    must_in_yaml = True

    def get_ops(self, args):
        op_dict = {}
        for op in args.op:
            if isinstance(op, str):
                for item in args.op[op]:
                    if '_name' not in item:
                        continue
                    op_dict[item._name] = {'name': item.name, 'input': item.input}
                    for k in ['output', 'cuda_name', 'cuda_mapping', 'cpu_name', 'cpu_mapping', '_mapping']:
                        if k in item:
                            op_dict[item._name][k] = item[k]
            else:
                if '_name' not in op:
                    continue
                if isinstance(op._name, str):
                    op_dict[op._name] = {'name': op.name, 'input': op.input}
                    for k in ['output', 'cuda_name', 'cuda_mapping', 'cpu_name', 'cpu_mapping', '_mapping']:
                        if k in op:
                            op_dict[op._name][k] = op[k]
                else:
                    for item in op._name:
                        op_dict[item] = {'name': op.name, 'input': op.input}
                        for k in ['output', 'cuda_name', 'cuda_mapping', 'cpu_name', 'cpu_mapping', '_mapping']:
                            if k in op:
                                op_dict[item][k] = op[k]
        return op_dict

    def get_op(self, op, on_info, ops):
        # schemas = torch._C._jit_get_all_schemas()
        # for schema in schemas:
        #     print(schema)
        if op in ops:
            output = ops[op].get('output', None)
            inputs = []
            if isinstance(ops[op]['input'][-1], DictConfig) and '_tensor' not in ops[op]['input'][-1]:
                args = ops[op]['input'][:-1]
                kwargs = ops[op]['input'][-1]
                len_op_input = len(args) + len(kwargs)
                inputs.extend([(i, x) for i, x in enumerate(args)])
                inputs.extend(kwargs.items())
            else:
                args = ops[op]['input']
                len_op_input = len(args)
                inputs.extend([(i, x) for i, x in enumerate(args)])
                kwargs = None
            if output and kwargs:
                for i, k in enumerate(kwargs):
                    if k in output:
                        output[output.index(k)] = i
            if kwargs and ('output' in kwargs or 'out' in kwargs):
                pass
            elif len(on_info['inputs'][0]) > len_op_input:
                for i in range(len(on_info['inputs'])):
                    need_del = []
                    for j in range(len(on_info['inputs'][i])):
                        if on_info['inputs'][i][j]['name'] in ['out', 'output', 'dtype'] and op not in self.op_filter:
                            need_del.insert(0, j)
                            # break
                    for j in need_del:
                        print('del:', j, on_info['inputs'][i][j])
                        del on_info['inputs'][i][j]
            if len(on_info['inputs'][0]) != len_op_input:
                print('schema not matched:', ops[op]['name'], len(on_info['inputs'][0]), len_op_input)
            on_info['schema'] = inputs
            for k in ['cuda_name', 'cuda_mapping', 'cpu_name', 'cpu_mapping', '_mapping']:
                if k in ops[op]:
                    on_info[k] = ops[op][k]
            return ops[op]['name'], on_info, output
        elif self.must_in_yaml:
            return None, on_info, None
        try:
            Op = getattr(torch.ops.aten, on_info['aten_op_name'].replace('aten::', ''))
            Op = on_info['aten_op_name']
            for i in range(len(on_info['inputs'])):
                need_del = []
                for j in range(len(on_info['inputs'][i])):
                    if on_info['inputs'][i][j]['name'] in ['out', 'output'] and Op not in self.op_filter:
                        need_del.append(j)
                        break
                for j in need_del:
                    # print('del:', j, on_info['inputs'][i][j])
                    del on_info['inputs'][i][j]
        except:
            Op = None
        return Op, on_info, None

    def load(self, args):
        load_case_file = args.load_case_file
        # 从args中获取max_cases_per_op参数，如果没有则使用默认值30
        max_cases_per_op = getattr(args, 'max_cases_per_op', 1)
        ops = self.get_ops(args)
        for op, op_info in load_op_txt(load_case_file, max_cases_per_op).items():
            op_name, on_info, output = self.get_op(op, op_info, ops)
            if not op_name:
                print('Failed to add in case:', op, on_info['aten_op_name'])
                continue
            case = DictConfig({
                'name': op_name,
                'input': []
            })
            if output:
                case['output'] = output
            for k in ['cuda_name', 'cuda_mapping', 'cpu_name', 'cpu_mapping']:
                if k in on_info:
                    case[k] = on_info[k]
            kwargs = {}
            for inputs in op_info['inputs']:
                case['input'] = []
                print(op)
                val_mapping = {}
                need_real_val = {}
                real_val_check = True
                case_precision = set()
                for i, input in enumerate(inputs):
                    print('\t', i, ' csv:', input['name'], input)
                    schema_id = (on_info['_mapping'].index(i) if i in on_info['_mapping'] else -1) if '_mapping' in on_info  else i
                    if schema_id >= len(on_info['schema']):
                        schema_id = -1
                    if op in ops and 'schema' in on_info and isinstance(on_info['schema'], (list, ListConfig)) and schema_id != -1:
                        in_name, in_schema = on_info['schema'][schema_id]
                        print('\t   yaml:',  in_name, in_schema)
                    if input['args_type'] == 'Tensor':
                        if input['value'] == 'None' or input['value']['dtype'] == 'undef':
                            if self.config_as_kwargs:
                                kwargs[input['name']] = None
                            else:
                                case['input'].append(None)
                        else:
                            _tensor = {
                                '_tensor': input['value']['size'],
                                'precision': input['value']['dtype'],
                                'strides': input['value']['strides'],
                            }
                            if 'fp8' in _tensor['precision']:
                                _tensor['precision'] = _tensor['precision'].replace('fp8', '')
                            if in_schema and isinstance(in_schema, (dict, DictConfig)):
                                if '_tensor' in in_schema and len(_tensor['_tensor']) != len(in_schema['_tensor']):
                                    print('schema not matched:', op, len(_tensor['_tensor']), len(in_schema['_tensor']))
                                    real_val_check = False
                                for j, v in enumerate(_tensor['_tensor']):
                                    if '_val' in in_schema:
                                        pass
                                    elif j < len(in_schema['_tensor']):
                                        if in_schema['_tensor'][j] in val_mapping and v not in val_mapping[in_schema['_tensor'][j]]:
                                            val_mapping[in_schema['_tensor'][j]].append(v)
                                        else:
                                            val_mapping[in_schema['_tensor'][j]] = [v]
                                if '_val' in in_schema:
                                    need_real_val[i] = in_schema
                                    if 'precision' in in_schema:
                                        _tensor['precision'] = in_schema['precision']
                                if in_schema is None:
                                    _tensor = None
                                if 'distributions' in in_schema:
                                    _tensor['distributions'] = in_schema['distributions']
                                if 'ulp_dtype' in in_schema:
                                    _tensor['ulp_dtype'] = in_schema['ulp_dtype']
                                if 'precision' in in_schema and 'high' in in_schema:
                                    _tensor['high'] = in_schema['high']
                                    if 'low' in in_schema:
                                        _tensor['low'] = in_schema['low']
                                    if isinstance(in_schema['high'], str):
                                        need_real_val[i] = in_schema
                                    elif 'low' in in_schema and isinstance(in_schema['low'], str):
                                        need_real_val[i] = in_schema
                            case_precision.add(_tensor['precision'])
                            if self.config_as_kwargs:
                                kwargs[input['name']] = _tensor
                            else:
                                case['input'].append(_tensor)
                    elif input['args_type'] == 'str':
                        if isinstance(in_schema, (int, float)) or input['value'] == 'auto':
                            case['input'].append(in_schema)
                    else:
                        try:
                            if self.config_as_kwargs:
                                kwargs[input['name']] = eval(input['value'])
                            else:
                                case['input'].append(eval(input['value']))
                        except:
                            if self.config_as_kwargs:
                                kwargs[input['name']] = input['value']
                            else:
                                case['input'].append(input['value'])
                case['input'].append(kwargs)
                if need_real_val:
                    assert real_val_check
                    for i in need_real_val:
                        _val = []
                        if '_val' in need_real_val[i]:
                            for j in need_real_val[i]['_val']:
                                assert j in val_mapping
                                assert len(val_mapping[j]) == 1
                                _val.append(val_mapping[j][0])
                            print('real value:', i,  _val, need_real_val[i]['_val'], inputs[i])
                            case['input'][i]['_val'] = _val
                            del case['input'][i]['_tensor']
                        if 'high' in need_real_val[i] and isinstance(need_real_val[i]['high'], str):
                            case['input'][i]['high'] = eval(need_real_val[i]['high'], {k:v[0] for k, v in val_mapping.items()})
                            print('real value:', i,  case['input'][i]['high'], need_real_val[i]['high'], inputs[i])
                        if 'low' in need_real_val[i] and isinstance(need_real_val[i]['low'], str):
                            case['input'][i]['low'] = eval(need_real_val[i]['low'], {k:v[0] for k, v in val_mapping.items()})
                            print('real value:', i,  case['input'][i]['low'], need_real_val[i]['low'], inputs[i])
                for precision in ['bfloat16', 'Half', 'Float', 'Int']:
                    if precision in case_precision:
                        case['precision'] = precision
                        break
                if '_mapping' in on_info:
                    _tmp = case['input']
                    case['input'] = []
                    for i, k in enumerate(on_info['_mapping']):
                        if k == -1:
                            case['input'].append(None)
                        else:
                            case['input'].append(_tmp[k])
                    case['input'].append(DictConfig({}))
                yield deepcopy(case)


case_loader = Loader()

if __name__ == '__main__':
    path = 'launcher/tools/operator_info.csv'
    op_dict = load_op_txt(path, max_cases_per_op=2000)  # 测试时也使用30组数据限制
    aten_op_num = 0
    customer_op = []
    for op, on_info in op_dict.items():
        try:
            Op = getattr(torch.ops.aten, on_info['aten_op_name'].replace('aten::', ''))
            aten_op_num += 1
        except:
            Op = None
            customer_op.append(op)
        print(Op, op, len(on_info['inputs']))
    print('aten_op_num', aten_op_num, 'customer_op', len(customer_op))
    print(customer_op)