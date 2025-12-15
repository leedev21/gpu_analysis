import os
import json
import torch
from torch.utils.data import Dataset
from moprobe.utils import loop_process_data, to_device
from launcher.utils import each_file

# def get_dtype(args):
#     if args in [32, 'fp32']:
#         return torch.float
#     if args in [16, 'fp16']:
#         return torch.half
#     if args in ['bf16']:
#         return torch.bfloat16
#     return None


def get_dtype(test_case='', args='', precision=''):
    if not precision:
        precision = test_case.precision if 'precision' in test_case else args.training.precision
    precision_map = {'fp32': torch.float,
                        'Float': torch.float,
                        32: torch.float,
                        'fp16': torch.half,
                        'half': torch.half,
                        16: torch.half,
                        'bf16': torch.bfloat16,
                        'bfloat16': torch.bfloat16,
                        'e4m3': torch.float8_e4m3fn,
                        'e5m2': torch.float8_e5m2,
                        }
    if precision in precision_map:
        return precision_map[precision]
    else:
        return None

def generate_tensor_with_stats(data, dtype, device, args, seed):
    if args.distribution == 'normal':
        tensor = torch.randn(data, generator=torch.Generator().manual_seed(seed))
        tensor = tensor * (2**args.mean)
        tensor = tensor * args.std
        max_val = 50 if args.max > 50 else args.max
        min_val = 50 if args.max > 50 else args.max
        tensor = torch.clamp(tensor, -2**min_val, 2**max_val)
        return tensor.to(dtype).to(device)
    elif args.distribution == 'uniform':
        # tensor = tensor * (2**args.mean)
        max_val = 50 if args.max > 50 else args.max
        min_val = 50 if args.max > 50 else args.max
        tensor = torch.Tensor(data).uniform_(-2**min_val, 2**max_val)
        # tensor = torch.clamp(tensor, -2**min_val, 2**max_val)
        return tensor.to(dtype).to(device)
    elif args.distribution == 'zeros':
        return torch.zeros(data, dtype=dtype).to(device)
    elif args.distribution == 'ones':
        return torch.ones(data, dtype=dtype).to(device)
    elif args.distribution == 'eye':
        return torch.eye(data, dtype=dtype).to(device)
    elif args.distribution == 'randint':
        return torch.randint(low=args.low, high=args.high, size=data, dtype=dtype).to(device)
    else:
        print('Err distribution not supported:', args.distribution)


class PTLoader():
    def __init__(self, skip_init=False, max_n=-1, pt_index=''):
        self.path = 'data/pt'
        self.skip_init = skip_init
        self.max_n = max_n
        self.pt_index = pt_index
        self.config = {}
        self.data = {}
        self.index = {0: {}}
        self.max_data = 10000
        self.module_info = {}
        self.sort_map = {}

    def _load_config(self, path):
        path = os.path.join(path, 'config') if path else os.path.join(self.path, 'config')
        if not os.path.exists(path):
            return
        print('init config found!')
        for file_name, file_path in each_file(path, endwith='.pt', with_file=True):
            config_name = file_name[file_name.rfind('init_')+5:-3]
            # print(config_name, file_path)
            try:
                self.config[config_name] = torch.load(file_path, weights_only=False)
                print(self.config[config_name])
            except:
                print('load config error:', file_path)

    def _load_init(self, task, filters):
        module_json = None
        def load_by_module_json(path):
            for file_name, file_path in each_file(path, endwith='module.json', with_file=True):
                split_name = file_path.split('/')
                try:
                    rank = int(split_name[-2])
                except:
                    rank = ''
                with open(file_path) as f:
                    for k, v in json.load(f).items():
                        k = k.replace('module::', '')
                        if filters is None or k in filters:
                            if k not in self.data:
                                self.data[k] = {}
                            self.data[k][rank] = {}
                            for case in v:
                                for layer in case['state']:
                                    layer = layer.split(':')[1].replace('_' + k, '')
                                    self.data[k][rank][layer] = {'init': [], 'call': [], 'return': []}

        def load_by_path(path):
            for file_name, file_path in each_file(path, endwith='pt', with_file=True):
                split_name = file_path.split('/')
                # rank = split_name[-2]
                split_name = file_name.split('_')
                rank = int(split_name[-2].replace('rank', ''))
                if split_name[-3] == 'init':
                    state = 'init'
                    module_cls = split_name[-4]
                    module_val = '_'.join(split_name[0:-4])
                    if filters is not None and module_cls not in filters:
                        continue
                    if module_cls not in self.data:
                        self.data[module_cls] = {}
                    if rank not in self.data[module_cls]:
                        self.data[module_cls][rank] = {}
                    if module_val not in self.data[module_cls][rank]:
                        self.data[module_cls][rank][module_val] = {'init': [], 'call': [], 'return': []}
                    self.data[module_cls][rank][module_val]['init'].append(file_path)

        init_path = os.path.join(self.path, 'init')
        if os.path.exists(init_path):
            load_by_path(init_path)
            print('load init:', init_path)
        else:
            if os.path.exists(os.path.join(self.path, f'run/{task}')):
                module_json = os.path.join(self.path, f'run/{task}')
            if os.path.exists(os.path.join(self.path, f'{task}')):
                module_json = os.path.join(self.path, f'{task}')
            print('load init:', module_json)
            if module_json:
                load_by_module_json(module_json)

    def each_file(self, run_path):
        pt_loaded = False
        if self.pt_index and os.path.exists(self.pt_index):
            with open(self.pt_index) as f:
                for file_path in f.readlines():
                    file_path = file_path.strip()
                    file_name = file_path.split('/')[-1]
                    pt_loaded = True
                    yield file_name, file_path
        if not pt_loaded:
            for file_name, file_path in each_file(run_path, endwith='trace.pt', with_file=True):
                yield file_name, file_path

    def _load_all(self, path, filters, _map):
        run_path = os.path.join(self.path, 'run')
        if not os.path.exists(run_path):
            if os.path.exists(os.path.join(self.path, f'run/{path}')):
                run_path = os.path.join(self.path, f'run/{path}')
            elif os.path.exists(os.path.join(self.path, f'{path}')):
                run_path = os.path.join(self.path, f'{path}')
            else:
                print('run path not found!', run_path)
                run_path = path
        for file_name, file_path in self.each_file(run_path):
            if not os.path.exists(file_path):
                print(file_name, 'not found!')
                continue
            print('file_name:', file_name)
            if file_name == 'model_trace.pt':
                continue
            split_name = file_name.split('_')
            if split_name[-3] != 'init':
                if self.max_n != -1 and int(split_name[-3].replace('n', '')) > self.max_n:
                    continue
            rank = int(split_name[-2].replace('rank', ''))
            module_cls = split_name[-4] if split_name[-3] == 'init' else split_name[-5]
            module_cls_first = False
            if filters is not None and module_cls not in filters:
                continue
            if self.skip_init:
                if module_cls not in self.data:
                    self.data[module_cls] = {}
                if rank not in self.data[module_cls]:
                    self.data[module_cls][rank] = {}
            else:
                if module_cls not in self.data:
                    module_cls = split_name[0]
                    module_cls_first = True
                    if module_cls not in self.data:
                        print('Err split:', module_cls, split_name)
                if rank not in self.data[module_cls]:
                    rank = ''
            print(module_cls, rank, module_cls_first, split_name[-4], split_name)
            if split_name[-3] == 'init':
                state = 'init'
                if not module_cls_first:
                    module_val = '_'.join(split_name[0:-4])
                else:
                    module_val = '_'.join(split_name[1:-3])
                if self.skip_init and module_val not in self.data[module_cls][rank]:
                    self.data[module_cls][rank][module_val] = {'init': [], 'call': [], 'return': []}
                self.data[module_cls][rank][module_val]['init'].append(file_path)
            else:
                n = int(split_name[-3].replace('n', ''))
                state = split_name[-4]
                iter = 0
                if not module_cls_first:
                    module_val = '_'.join(split_name[:-5])
                else:
                    module_val = '_'.join(split_name[1:-4])
                module_name_split = module_val.split('.')
                module_name = '.'.join(module_name_split[1:])

                if _map:
                    if _map['layer'] in module_val:
                        module_name = module_name.replace(_map['layer']+'.', '')
                        module_name_split = module_name.split('.')
                        n_layer = int(module_name_split[0])
                        module_name = '.'.join(module_name_split[1:])
                    else:
                        if module_name in _map['post']:
                            n_layer = 'post'
                        else:
                            n_layer = -1
                    # print('check', module_val, iter, n_layer, module_name, _map, split_name)
                if self.skip_init and module_val not in self.data[module_cls][rank]:
                    self.data[module_cls][rank][module_val] = {'init': [], 'call': [], 'return': []}
                self.data[module_cls][rank][module_val][state].append((n, file_path))
                if _map:
                    if rank not in self.index[0]:
                        self.index[0][rank] = {}
                    if n_layer not in self.index[0][rank]:
                        self.index[0][rank][n_layer] = {}
                    if module_name not in self.index[0][rank][n_layer]:
                        self.index[0][rank][n_layer][module_name] = {}
                    if state not in self.index[0][rank][n_layer][module_name]:
                        self.index[0][rank][n_layer][module_name][state] = []
                    if module_cls not in self.module_info:
                        self.module_info[module_cls] = {}
                    if n_layer not in self.module_info[module_cls]:
                        self.module_info[module_cls][n_layer] = {}
                    if module_name not in self.module_info[module_cls][n_layer]:
                        self.module_info[module_cls][n_layer][module_name] = set()
                    self.module_info[module_cls][n_layer][module_name].add(module_val)
                    self.index[0][rank][n_layer][module_name][state].append((n, file_path))
        print('module test load from file:')
        for module_cls in self.data:
            for rank in self.data[module_cls]:
                # print(module_cls, rank)
                for module_val in self.data[module_cls][rank]:
                    print('\t', module_val, self.data[module_cls][rank][module_val].keys())
                    for state in self.data[module_cls][rank][module_val]:
                        if state != 'init':
                            self.data[module_cls][rank][module_val][state] = list(sorted(self.data[module_cls][rank][module_val][state], key=lambda x: x[0]))
                            if len(self.data[module_cls][rank][module_val][state]) > self.max_data:
                                self.data[module_cls][rank][module_val][state] = self.data[module_cls][rank][module_val][state][0:self.max_data]
                        # for path in self.data[module_cls][rank][module_val][state]:
                        #     print('\t\t', state, path)

        if _map:
            ranks = set()
            n_layers = set()
            for request in self.index:
                for rank in self.index[request]:
                    ranks = ranks.union(self.index[request].keys())
                    for n_layer in self.index[request][rank]:
                        n_layers = n_layers.union(self.index[request][rank].keys())
                        for module_name in self.index[request][rank][n_layer]:
                            for state in ['call', 'return']:
                                if state not in self.index[request][rank][n_layer][module_name]:
                                    continue
                                print('index add', request, rank, n_layer, module_name, state)
                                self.index[request][rank][n_layer][module_name][state] = list(sorted(self.index[request][rank][n_layer][module_name][state], key=lambda x: x[0]))
            self.sort_map['request'] = sorted(self.index.keys())
            self.sort_map['rank'] = sorted(ranks)
            self.sort_map['n_layer'] = sorted(n_layers, key=lambda x: 1000 if isinstance(x, str) else x)
            self.sort_map['module_name'] = [items.split(':') for items in _map['transformer']]
            for k, v in self.sort_map.items():
                print('load:', k, v)

    def load(self, task, filters, path='', map={}):
        self.path = path if path else os.path.join(self.path, task)
        if not self.data:
            self._load_config(task if '/' in task else '')
            if not self.skip_init:
                self._load_init(task, filters)
            self._load_all(os.path.join(self.path, task) if not path else task, filters, map)

    def get(self, module_cls):
        if module_cls in self.data:
            rank = list(self.data[module_cls].keys())[0]
            for module_val in self.data[module_cls][rank]:
                if 'call' in self.data[module_cls][rank][module_val] and 'return' in self.data[module_cls][rank][module_val]:
                    yield self.data[module_cls][rank][module_val]['init'], list(zip(self.data[module_cls][rank][module_val]['call'], self.data[module_cls][rank][module_val]['return']))

    def get_by_module_cls(self, request, rank, n_layer, module_cls, _iter=None):
        if module_cls in self.data:
            for module_name in self.module_info[module_cls][n_layer]:
                for module_val in self.module_info[module_cls][n_layer][module_name]:
                    if module_name in self.index[request][rank][n_layer]:
                        if _iter is not None:
                            yield self.data[module_cls][rank][module_val]['init'], [(self.index[request][rank][n_layer][module_name]['call'][_iter], self.index[request][rank][n_layer][module_name]['return'][_iter])]
                        else:
                            yield self.data[module_cls][rank][module_val]['init'], list(zip(self.index[request][rank][n_layer][module_name]['call'], self.index[request][rank][n_layer][module_name]['return']))

    def for_each(self):
        for request in self.sort_map['request']:
            if request not in self.index:
                continue
            for rank in self.sort_map['rank']:
                if rank not in self.index[request]:
                    continue
                for n_layer in self.sort_map['n_layer']:
                    if n_layer not in self.index[request][rank]:
                        continue
                    for items in self.sort_map['module_name']:
                        if len(items) == 1:
                            module_name, state = items[0], None
                        else:
                            module_name, state = items
                        if module_name not in self.index[request][rank][n_layer]:
                            continue
                        print(request, rank, n_layer, module_name, state)
                        if state:
                    # for module_name in self.index[request][rank][n_layer]:
                        # for state in ['call', 'return']:
                            if state not in self.index[request][rank][n_layer][module_name]:
                                continue
                            for i in range(len(self.index[request][rank][n_layer][module_name][state])):
                                yield request, rank, i, n_layer, module_name, state
                        else:
                            for state in ['call', 'return']:
                                if state not in self.index[request][rank][n_layer][module_name]:
                                    continue
                                for i in range(len(self.index[request][rank][n_layer][module_name][state])):
                                    yield request, rank, i, n_layer, module_name, state


    def read_data(self, request, rank, i, n_layer, module_name, state):
        if i >= len(self.index[request][rank][n_layer][module_name][state]):
            print('Err index:', request, rank, i, n_layer, module_name, state, len(self.index[request][rank][n_layer][module_name][state]))
            return None
        path = self.index[request][rank][n_layer][module_name][state][i][1]
        # print(path)
        try:
            data = torch.load(path, weights_only=False, map_location='cpu')
        except:
            data = {'inputs': [], 'outputs': []}
        if state == 'call':
            return data['inputs']
        if state == 'return':
            return data['outputs']

    def read_and_merge_data(self, request, i, n_layer, module_name, state, split=None):
        tensor_shape = []
        target_tensor = []
        rank = 0
        # print('read_and_merge_data:', request, i, n_layer, module_name, state)
        # print('request', request in self.index)
        # print('rank', rank in self.index[request])
        # print('n_layer', n_layer in self.index[request][rank])
        if i >= len(self.index[request][rank][n_layer][module_name][state]):
            print('Err index:', request, rank, i, n_layer, module_name, state, len(self.index[request][rank][n_layer][module_name][state]))
            return [], []

        def get_tensor_shape_by_loop_process(data):
            def func(device, data):
                if isinstance(data, torch.Tensor):
                    return data.shape
                return data
            return loop_process_data(None, data, func)

        def split_tensor_by_loop_process(data):
            def func(device, data):
                if isinstance(data, torch.Tensor):
                    if split['type'] == 'batch_split':
                        batch = data.shape[0]
                        start = int(batch/split['k'] * split['i'])
                        end = int(batch/split['k'] * (split['i'] + 1))
                        return data[start:end]
                    else:
                        return data
                return data
            return loop_process_data(None, data, func)

        for rank in self.index[request]:
            _data = self.read_data(request, rank, i, n_layer, module_name, state)
            if split:
                _data = split_tensor_by_loop_process(_data)
            tensor_shape.append(get_tensor_shape_by_loop_process(_data))
            # try:
            # print('*'*100)
            # print(_data)
            # print('*'*100)

            if isinstance(_data, torch.Tensor) or len(_data) == 0:
                target_tensor.append(_data)
            elif len(_data[0]) > 0:
                target_tensor.append(_data[0][0])
            else:
                # print('*'*100)
                # print(_data)
                # print('*'*100)
                try:
                    target_tensor.extend(_data[1].values())
                except:
                    pass
            # except:
            #     print(_tensor)
            #     exit()
        return tensor_shape, target_tensor

pt_loader = PTLoader()


class RandomDataset(Dataset):
    def __init__(self, shape_list, n_samples, dtype=torch.half, distribution=None):
        self.len = n_samples
        self.data = []
        self.type = 'randn'
        self.distribution = distribution
        self.new(shape_list, dtype)
        ### need update

    def __getitem__(self, index):
        # items = [self.data[k][index] for k in range(len(self.data))]
        return self.data[index]

    def __len__(self):
        return self.len

    def new(self, shape_list, dtype):
        for shape in shape_list:
            shape = list(shape)
            print('load data:', shape)
            if isinstance(shape[0], str):
                if shape[0] == 'emb':
                    random_floats = torch.rand(self.len, shape[1], 1) * shape[2]
                    _tensor = random_floats.reshape(self.len, -1).to(torch.int64)
            elif self.distribution:
                shape.insert(0, self.len)
                _tensor = generate_tensor_with_stats(shape, dtype=dtype, device='cpu', args=self.distribution)
            else:
                shape.insert(0, self.len)
                _tensor = torch.randn(shape, dtype=dtype)
            self.data.append(_tensor)

    def next_iter(self):
        if len(self.data) == 0:
            return None
        return self.data[0]

    def to(self, device):
        self.data = to_device(device, self.data)