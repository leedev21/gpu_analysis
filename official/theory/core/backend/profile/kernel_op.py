import os
import re
from theory.core import utils
from theory.core.utils import collection_path


class KernelOpMap(object):
    def __init__(self):
        self.tags = ['op_name', 'kernel_name', 'big_op', 'regex', 'hw_name', 'op_precision', 'op_group', 'op_type']
        self.map = {}
        self.aten = {}
        self.op = {}
        self.layer = {}
        self.base_config_path = os.path.join(collection_path(), 'config/ops')
        self.read_map(utils.read_file(self.base_config_path, 'kernel_op.map'))
        self.read_aten_map(utils.read_file(self.base_config_path, 'aten_op.map'))
        self.read_map(utils.read_file(self.base_config_path, 'user_annotation.map'))
        self.read_map(utils.read_file(self.base_config_path, 'launch_kernel.map'))

    def regist_key(self, name, tag, value):
        if tag not in self.map:
            self.map[tag] = {}
        self.map[tag][name] = value

    def read_map(self, data, sep='\t'):
        for line in data:
            items = line.strip().split(sep)
            if len(items) == 1 and items[0] == '':
                continue
            if len(items) != 3:
                print(len(items), items)
                input('Err')
            self.regist_key(items[0], items[1], items[2])

    def regist_aten(self, aten_name, op_list, op_type):
        # print(aten_name, op_list, op_type)
        if aten_name not in self.aten:
            self.aten[aten_name] = {'op_type': op_type, 'op': op_list}
        for op in op_list:
            if op.startswith('OP'):
                if op not in self.op:
                    self.op[op] = [aten_name]
                else:
                    self.op[op].append(aten_name)

    def regist_aten_list(self, aten_name, op_info, op_type, split_s):
        op_list = op_info.split(split_s)
        for op in op_list:
            if split_s == ',':
                self.regist_aten_list(aten_name, op, op_type, ';')
            if split_s == ';':
                op_split = op.split('::')
                if op_split[0] not in ['OP', 'aten', 'None']:
                    print(aten_name, op_list, op_type)
                    input('Err')
                self.regist_aten(aten_name, op_list, op_type)

    def read_aten_map(self, data, sep='\t'):
        for line in data:
            items = line.strip().split(sep)
            if len(items) == 1 and items[0] == '':
                continue
            if len(items) not in [4, 5]:
                print(len(items), items)
                input('Err')
            if items[3] not in ['None', '2D', '1D', 'RNG', 'SFU']:
                pass
                # print(len(items), items)
                # input('Err')
            self.regist_aten_list(items[0], items[2], items[3], ',')
        for aten in self.aten:
            for op in self.aten[aten]['op']:
                if op.startswith('aten') and op not in self.aten:
                    print(aten, op)
                    input('Err')
        for op in self.op:
            if any(op in k and op !=k for k in self.op):
                print(op)
                # input('Err')

    def regex(self, tag, line):
        for key in self.map[tag]:
            index = re.search(key, line)
            # if 'CudaCodeGen::kernel1' in line:
            #     print(index, key, line)
            #     input('Pause')
            if index != None:
                name = index.group().replace('(', '').replace(')', '')
                return {'op_name': name, 'kernel_name': name}
        return {}

    def search(self, tag, line):
        res = {tag: [], 'kernel_name': [], 'op_name': []}
        for key in self.map[tag]:
            index = re.search(re.escape(key), line)
            if index != None:
                # print('index:', index)
                val = self.map[tag][key]
                if tag in ['op_name', 'kernel_name']:
                    if all(key not in k for k in res['kernel_name']):
                        res['kernel_name'].append(key)
                if '{' in val:
                    val = eval(val)
                    if tag != 'kernel_name':
                        res[tag] = val
                    for k in ['op_name', 'kernel_name']:
                        if k in val and all(val[k] not in a for a in res[k]):
                            res[k].append(val[k])
                elif all(val not in k for k in res[tag]):
                    res[tag].append(val)
        if res[tag]:
            return res
        return {}

    def match(self, line):
        res = {'kernel_name': []}
        for tag in self.tags:
            if hasattr(self, tag):
                for k, v in getattr(self, tag)(tag, line).items():
                    # print('\tgetattr:', tag, k, v, res)
                    if isinstance(v, list):
                        if k in res:
                            res[k].extend(v)
                        else:
                            res[k] = v
                    else:
                        if k in res:
                            res[k].append(v)
                        else:
                            res[k] = [v]
            else:
                for k, v in self.search(tag, line).items():
                    # print('\tsearch:', tag, k, v, res)
                    if isinstance(v, list):
                        if k in res:
                            res[k].extend(v)
                        else:
                            res[k] = v
                    else:
                        if k in res:
                            res[k].append(v)
                        else:
                            res[k] = [v]
        return res


class KernelOp(object):
    def __init__(self, _str='', fake=False):
        self.op_group = ''
        self.op_name_list = []
        self.kernel_name_list = []
        self.op_type = ''
        self.op_precision = ''
        self.op_name = None
        self.kernel_name = ''
        self.res = ''
        self.op_n = 0
        self.map = KernelOpMap()
        if fake:
            self.create_fake_op(_str)
        else:
            self.create_op(_str)

    def set(self, k, v):
        # print(k, v, self.op_name_list)
        if hasattr(self, k):
            setattr(self, k, v)
        if hasattr(self, k + '_list'):
            if isinstance(v, str):
                if v in self.op_name_list or v in self.kernel_name_list:
                    return
                else:
                    getattr(self, k + '_list').append(v)
            getattr(self, k + '_list').extend(v)

    def create_fake_op(self, _str):
        if not _str:
            return
        self.op_name = _str

    def create_op(self, _str):
        if not _str:
            return
        if _str.startswith('aten::'):
            self.op_name = _str
            self.op_group = 'aten'
            self.op_type = ''
            return
        self.res = self.map.match(_str)
        if not self.res.get('op_name'):
            print('res:', _str, self.res)
            print('no op_name in res')
            exit()
        if not self.res:
            self.op_name = _str
            self.op_group = 'check'
            self.op_type = ''
            return
        for k, v in self.res.items():
            self.set(k, v)
        if 'big_op' in self.res:
            for k, v in self.res['big_op'][0].items():
                #print(k, v)
                if k == 'type':
                    k = 'op_type'
                self.set(k, v)
        if isinstance(self.op_name, list):
            self.op_name = ':'.join(self.op_name)
        if isinstance(self.kernel_name, list):
            tmp = []
            for item in self.kernel_name:
                if all(item not in k for k in tmp):
                    tmp.append(item)
            self.kernel_name = ':'.join(self.kernel_name)
        if isinstance(self.op_group, list):
            self.op_group = ':'.join(self.op_group)
        if isinstance(self.op_type, list):
            self.op_type = ':'.join(self.op_type)
        if isinstance(self.op_precision, list):
            self.op_precision = ':'.join(self.op_precision)
        # if self.op_group not in ['Nccl'] and self.op_name not in ['Memset', 'epilogue:BgradAB', 'direct_copy', 'BUnary:Mul', 'AUnary:Mul', 'multi:L2Norm', 'cleanup', 'Reduce']:
        #     if len(self.op_name_list) > 1:
        #     # if any(k in self.op_name for k in ['AUnary']):
        #         print(_str)
        #         self.name()
        #         input('Next')

    def name(self):
        if len(self.op_name_list) > 1:
            name = 'OP::' + ':'.join(self.op_name_list)
        elif not self.op_name:
            return None, None
        elif self.op_name.startswith('aten::'):
            name = self.op_name
        elif self.op_name.startswith('OP::'):
            name = self.op_name
        else:
            name = 'OP::' + self.op_name
        if self.op_type in ['fwd', 'bwd']:
            name += ':' + self.op_type
        # print('==============' + name + '==============  ' + str(self.map.op.get(name)))
        if name not in self.map.op:
            # print(self.res)
            # print(self.op_name, name)
            # for k in self.map.op:
            #     print(k, self.map.op[k])
            return name, None
        return name, self.map.op.get(name)

    def get_kernel_name(self):
        if len(self.op_name_list) > 1:
            name = ':'.join(self.kernel_name_list)
        elif not self.kernel_name:
            return None
        else:
            name = self.kernel_name
        if self.op_type:
            name += ':' + self.op_type
        if self.op_precision:
            name += ':' + self.op_precision
        if self.op_group:
            name = self.op_group + ':' + name
        name = 'Kernel::' + name
        return name

    def show(self):
        print('==============' + self.name() + '==============')
        for item in self.map.tags:
            if item == 'op_name':
                continue
            if hasattr(self, item) and getattr(self, item):
                print(item, ':', getattr(self, item))
