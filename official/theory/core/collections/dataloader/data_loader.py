import torch
import torch.distributed
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from moprobe.utils import loop_process_data, to_device, get_dtype_to_str


configs = {
    'DEVICE': None,
    'IntTensor_zero': None
}


def clone_by_loop_process(data):
    def copy_to(device, data):
        return data.clone()
    return loop_process_data(None, data, copy_to)


def get_tensor_info_by_loop_process(device, data):
    def func(device, data):
        _info = {'dtype': get_dtype_to_str(data.dtype), 'shape': list(data.shape), 'device': data.device.type}
        if device:
            _info['k'] = device
        return _info
    def exception(device, data):
        try:
            _info = {'dtype': type(data), 'shape': [1], 'device': 'cpu'}
        except:
            _info = {'dtype': 'unknown', 'shape': [1], 'device': 'cpu'}
        if device:
            _info['k'] = device
        return _info
    return loop_process_data(device, data, func, exception)


class BatchSampler():
    def __init__(self, test_case, config, seed):
        self._args = []
        self._kwargs = {}
        self.args = []
        self.kwargs = {}
        self.test_case = test_case
        self.out_mapping = test_case.output
        self.len_kwargs = 0
        self.config = config
        self.seed = seed
        self.by_new_style = True
        self.scales = []
        self._out_info = []
        self.fp8_index = {}
        self.process_groups = {}
        self.create()

    @staticmethod
    def get_tenor_zero(size):
        global configs
        if configs['IntTensor_zero'] is None:
            configs['IntTensor_zero'] = torch.cuda.IntTensor(size).zero_()
        return configs['IntTensor_zero']

    @staticmethod
    def get_dtype(precision='', test_case=''):
        if not precision:
            precision = test_case.precision
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
        elif isinstance(precision, str) and precision.lower() in precision_map:
            return precision_map[precision.lower()]
        else:
            return None

    # @staticmethod
    def generate_tensor_with_stats(self, data, dtype, device, args, k_tensor=None):
        if args.distribution == 'normal':
            tensor = torch.randn(data)
            # tensor = torch.randn(data, generator=torch.Generator().manual_seed(seed))
            tensor = tensor * (2**args.mean)
            tensor = tensor * args.std
            max_val = 50 if args.max > 50 else args.max
            min_val = 50 if args.max > 50 else args.max
            tensor = torch.clamp(tensor, -2**min_val, 2**max_val)
            if dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                from launcher.runner.module.layers.quant import native_per_token_group_quant_fp8, per_block_cast_to_fp8
                if not self.scales:
                    x_q, x_s = native_per_token_group_quant_fp8(tensor)
                else:
                    x_q, x_s = per_block_cast_to_fp8(tensor)
                self.scales.append(x_s)
                if torch.distributed.is_initialized():
                    if k_tensor is not None:
                        self.fp8_index[k_tensor] = dtype
                    dtype = torch.float
                # finfo = torch.finfo(dtype)
                # return tensor.clamp(min=finfo.min, max=finfo.max).to(dtype=dtype).to(device)
                return x_q.to(dtype=dtype).to(device)
            else:
                return tensor.to(dtype).to(device)
        elif args.distribution == 'uniform':
            # tensor = tensor * (2**args.mean)
            max_val = 50 if args.max > 50 else args.max
            min_val = 50 if args.max > 50 else args.max
            tensor = torch.Tensor(data).uniform_(-2**min_val, 2**max_val)
            # tensor = torch.clamp(tensor, -2**min_val, 2**max_val)
            return tensor.to(dtype).to(device)
        elif args.distribution == 'scale':
            x_s = self.scales.pop(0)
            # return torch.randn(data, dtype=dtype).t().contiguous().t().to(device)
            return x_s.to(dtype=dtype).to(device)
        elif args.distribution == 'scale_t':
            x_s = self.scales.pop(0)
            # return torch.randn(data, dtype=dtype).t().contiguous().t().to(device)
            return x_s.to(dtype=dtype).t().contiguous().t().to(device)
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

    def rand_tensor(self, data, dtype, device, args=None, strides=None, distributions=None, k_tensor=None):
        if isinstance(dtype, str):
            precision = self.get_dtype(precision=dtype)
        else:
            precision = dtype
        if distributions:
            return self.generate_tensor_with_stats(data, dtype=precision, device=device, args=distributions, k_tensor=k_tensor)
        elif isinstance(dtype, str):
            if dtype.lower() in ['long', 'int', 'int32', 'int64']:
                dtype = torch.long if dtype.lower() in ['long', 'int64'] else torch.int
                print('args:', args)
                return torch.randint(args.get("low", 0), args.get('high', 32768), data, dtype=dtype).to(device)
            elif dtype.lower() in ['bool']:
                _tensor = torch.FloatTensor(*data).uniform_() > 0.7
                return _tensor.to(device)
            elif dtype.lower() in ['e4m3', 'e5m2']:
                if torch.distributed.is_initialized():
                    if k_tensor is not None:
                        self.fp8_index[k_tensor] = dtype
                    precision = torch.float
                return torch.randn(data, dtype=torch.bfloat16).to(device).to(precision)
        return torch.randn(data, dtype=precision).to(device)

    def create_in_by_cls(self, cls, _in, dtype, device, distributions, k_tensor):
        def create_tensor_list(data, dtype, device, _in, strides, distribution):
            if isinstance(data[0], (ListConfig, list)):
                res = []
                if 'n_tensor' in _in:
                    for i in range(int(_in['n_tensor'])):
                        res.append(create_tensor_list(data[0], dtype, device, _in, strides, distribution))
                else:
                    for item in data:
                        res.append(create_tensor_list(item, dtype, device, _in, strides, distribution))
            else:
                return self.rand_tensor(list(data), dtype, device, _in, strides, distribution, k_tensor)
            return res
        if cls == '_tensor':
            if _in['_tensor'] == 'Zero':
                try:
                    self.args.append(self.get_tenor_zero(1))
                except:
                    pass
            else:
                _dtype = _in.precision if 'precision' in _in else dtype
                strides = _in.strides if 'strides' in _in else None
                _distribution = distributions[_in.distributions] if 'distributions' in _in else None
                return self.rand_tensor(list(_in['_tensor']), _dtype, device, _in, strides, _distribution, k_tensor)
        elif cls == '_tensor_list':
            _dtype = _in.precision if 'precision' in _in else dtype
            strides = _in.strides if 'strides' in _in else None
            _distribution = distributions[_in.distributions] if 'distributions' in _in else None
            return create_tensor_list(_in['_tensor_list'], _dtype, device, _in, strides, _distribution)
        elif cls == '_process_group':
            n_groups = torch.distributed.get_world_size() / _in['_process_group']
            group_id = torch.distributed.get_rank() // _in['_process_group']
            group_start = group_id * _in['_process_group']
            print('check:', n_groups, group_id, group_start, f"{n_groups}_{group_id}")
            if f"{n_groups}_{group_id}" not in self.process_groups:
                group_ranks = [i for i in range(_in['_process_group'])]
                self.process_groups[f"{n_groups}_{group_id}"] = torch.distributed.new_group(ranks=group_ranks, backend='nccl')
            return self.process_groups[f"{n_groups}_{group_id}"]
        elif cls == '_dist_red_op':
            return getattr(torch.distributed.ReduceOp, _in['_dist_red_op'])
        elif isinstance(_in, (DictConfig, dict)):
            _dtype = _in.precision if 'precision' in _in else dtype
            val_data = _in['_val']
            if isinstance(val_data, (ListConfig, list)):
                val_data = [v for v in val_data]
            return torch.tensor(val_data, dtype=self.get_dtype(_dtype), device=device)
        elif isinstance(_in, (ListConfig, list)):
            return [v for v in _in]
        elif isinstance(_in, str):
            try:
                return eval(_in)
            except:
                return _in
        elif isinstance(_in, (int, float, bool, type(None))):
            return _in
        else:
            print('Warning! not register:', _in, type(_in))
            return _in

    def random_data_new(self, inputs, dtype, device, config):
        distributions = config['distributions']
        for _in in inputs:
            if isinstance(_in, (DictConfig, dict)):
                data_type = '_dict'
                for cls in ['_tensor', '_tensor_list', '_val', '_process_group', '_dist_red_op']:
                    if cls in _in:
                        data_type  = cls
                        self.args.append(self.create_in_by_cls(data_type, _in, dtype, device, distributions, len(self.args)))
                if data_type == '_dict':
                    for k in _in:
                        data_type = '_val'
                        if isinstance(_in[k], (DictConfig, dict)):
                            for cls in ['_tensor', '_tensor_list', '_val', '_process_group', '_dist_red_op']:
                                if cls in _in[k]:
                                    data_type = cls
                                    break
                        self.kwargs[k] = self.create_in_by_cls(data_type, _in[k], dtype, device, distributions, k)
            elif isinstance(_in, (ListConfig, list)):
                self.args.append(self.create_in_by_cls('_tensor_list', _in, dtype, device, distributions, len(self.args)))
            else:
                self.args.append(self.create_in_by_cls('_val', _in, dtype, device, distributions, len(self.args)))
        self.len_kwargs = len(self.kwargs)

    def random_data(self, inputs, dtype, device, config):
        def create_tensor_list(data):
            if isinstance(data[0], ListConfig):
                res = []
                for item in data:
                    res.append(create_tensor_list(item))
            else:
                return self.rand_tensor(list(data), dtype, device)
            return res
        for _in in inputs:
            if isinstance(_in, (int, float, bool, type(None))):
                self.args.append(_in)
            elif isinstance(_in, (ListConfig, list)):
                self.args.append(create_tensor_list(_in))
            elif '_tensor' in _in:
                if _in['_tensor'] == 'Zero':
                    try:
                        self.args.append(self.get_tenor_zero(1))
                    except:
                        pass
                else:
                    _dypte = _in.precision if 'precision' in _in else dtype
                    strides = _in.strides if 'strides' in _in else None
                    distributions = config['distributions'][_in.distributions] if 'distributions' in _in else None
                    self.args.append(self.rand_tensor(list(_in['_tensor']), _dypte, device, _in, strides, distributions))
            elif '_tensor_list' in _in:
                # Handle tensor list generation
                print(f"Processing _tensor_list: {_in}")
                _dypte = _in.precision if 'precision' in _in else dtype
                tensor_list = []
                for tensor_shape in _in['_tensor_list']:
                    tensor_list.append(self.rand_tensor(list(tensor_shape), _dypte, device, _in))
                print(f"Generated tensor_list with {len(tensor_list)} tensors, shapes: {[t.shape for t in tensor_list]}")
                self.args.append(tensor_list)
                # Process other keys in the dict as kwargs
                for k in _in:
                    if k not in ['_tensor_list', 'precision']:
                        self.kwargs[k] = _in[k]
            elif isinstance(_in, (DictConfig, dict)):
                data_type = '_val'
                # Special handling for _tensor_list at dict level
                if '_tensor_list' in _in:
                    print(f"Processing _tensor_list: {_in}")
                    _dypte = _in.precision if 'precision' in _in else dtype
                    tensor_list = []
                    for tensor_shape in _in['_tensor_list']:
                        tensor_list.append(self.rand_tensor(list(tensor_shape), _dypte, device, _in))
                    print(f"Generated tensor_list with {len(tensor_list)} tensors, shapes: {[t.shape for t in tensor_list]}")
                    self.args.append(tensor_list)
                    # Process other keys in the dict as kwargs
                    for k in _in:
                        if k not in ['_tensor_list', 'precision']:
                            self.kwargs[k] = _in[k]
                else:
                    # Normal dict processing
                    for k in _in:
                        if k == data_type:
                            if isinstance(_in[k], ListConfig):
                                self.args.append([v for v in _in[k]])
                            elif isinstance(_in[k], str):
                                self.args.append(eval(_in[k]))
                            else:
                                self.args.append(_in[k])
                        elif isinstance(_in[k], DictConfig):
                            if '_tensor' in _in[k]:
                                data = list(_in[k]['_tensor'])
                                _dypte = _in[k].precision if 'precision' in _in[k] else dtype
                                strides = _in.strides if 'strides' in _in[k] else None
                                distributions = config['distributions'][_in[k].distributions] if 'distributions' in _in[k] else None
                                self.kwargs[k] = self.rand_tensor(data, _dypte, device, _in[k], strides, distributions)
                            elif '_tensor_list' in _in[k]:
                                # Handle nested _tensor_list
                                print(f"Processing nested _tensor_list for key '{k}': {_in[k]}")
                                _dypte = _in[k].precision if 'precision' in _in[k] else dtype
                                tensor_list = []
                                for tensor_shape in _in[k]['_tensor_list']:
                                    if k == 'outputs':
                                        # For outputs parameter, create zero tensors (in-place operation)
                                        tensor_list.append(torch.zeros(list(tensor_shape), dtype=self.get_dtype(precision=_dypte), device=device))
                                    else:
                                        # For other parameters, create random tensors
                                        tensor_list.append(self.rand_tensor(list(tensor_shape), _dypte, device, _in[k]))
                                print(f"Generated tensor_list for '{k}' with {len(tensor_list)} tensors, shapes: {[t.shape for t in tensor_list]}")
                                self.kwargs[k] = tensor_list
                            elif '_val' in _in[k]:
                                # Handle nested _val (convert list to tensor)
                                print(f"Processing nested _val for key '{k}': {_in[k]}")
                                _dypte = _in[k].precision if 'precision' in _in[k] else dtype
                                val_data = _in[k]['_val']
                                if isinstance(val_data, ListConfig):
                                    val_data = [v for v in val_data]
                                # Convert to tensor with proper dtype
                                self.kwargs[k] = torch.tensor(val_data, dtype=self.get_dtype(precision=_dypte), device=device)
                                print(f"Generated tensor for '{k}': {self.kwargs[k]}")
                        else:
                            self.kwargs[k] = _in[k]
            else:
                print('Warning:', _in, type(_in))
                self.args.append(_in)

        # Special handling for moe_align_block_size operator
        if hasattr(self, 'case') and hasattr(self.case, 'name') and ('moe_align_block_size' in str(self.case.name) or 'exts_moe_align_block_size' in str(self.case.name)):
            print("Applying special handling for moe_align_block_size operator")
            if 'topk_ids' in self.kwargs and 'num_experts' in self.kwargs and 'block_size' in self.kwargs:
                topk_ids = self.kwargs['topk_ids']
                num_experts = self.kwargs['num_experts']
                block_size = self.kwargs['block_size']

                # Calculate token distribution and required output sizes
                token_num = topk_ids.size(0)
                topk = topk_ids.size(1)

                # Count tokens per expert
                expert_token_cnt = [0] * num_experts
                topk_ids_cpu = topk_ids.cpu()
                for i in range(token_num):
                    for j in range(topk):
                        expert = topk_ids_cpu[i][j].item()
                        if expert < num_experts:
                            expert_token_cnt[expert] += 1

                # Calculate padded sizes
                padded_cumsum = [0] * (num_experts + 1)
                for i in range(num_experts):
                    padded_cumsum[i + 1] = padded_cumsum[i] + ((expert_token_cnt[i] + block_size - 1) // block_size) * block_size

                total_tokens_padded = padded_cumsum[num_experts]
                expert_ids_size = total_tokens_padded // block_size if block_size > 0 and total_tokens_padded > 0 else 0

                print(f"Calculated sizes: total_tokens_padded={total_tokens_padded}, expert_ids_size={expert_ids_size}")

                # Only resize if calculated size is larger than existing size to avoid runtime errors
                if 'sorted_token_ids' in self.kwargs:
                    current_size = self.kwargs['sorted_token_ids'].size(0)
                    if total_tokens_padded > current_size:
                        print(f"Resizing sorted_token_ids from {current_size} to {total_tokens_padded}")
                        self.kwargs['sorted_token_ids'] = torch.zeros(total_tokens_padded, dtype=torch.int64, device=device)
                    else:
                        print(f"Keeping original sorted_token_ids size {current_size} (calculated: {total_tokens_padded})")

                if 'experts_ids' in self.kwargs:
                    current_size = self.kwargs['experts_ids'].size(0)
                    if expert_ids_size > current_size:
                        print(f"Resizing experts_ids from {current_size} to {expert_ids_size}")
                        self.kwargs['experts_ids'] = torch.zeros(expert_ids_size, dtype=torch.int32, device=device)
                    else:
                        print(f"Keeping original experts_ids size {current_size} (calculated: {expert_ids_size})")

                if 'num_tokens_post_pad' in self.kwargs:
                    # This one is always size 1, so keep as is
                    pass
        self.len_kwargs = len(self.kwargs)

    def by_list(self):
        for k, v in self.kwargs.items():
            if k in self.out_mapping:
                self.out_mapping[self.out_mapping.index(k)] = len(self.args)
            self.args.append(v)
        self.len_kwargs = 0

    def print_config(self):
        print('BatchSampler:', self.test_case, self.config, self.seed)

    def print_input(self):
        print('---------------------- input -----------------------')
        for i, v in enumerate(self._args):
            if isinstance(v, torch.Tensor):
                print(i, v.device, v.shape, v.dtype)
            else:
                print(i, v)
        if self.has_kwargs():
            for k, v in self._kwargs.items():
                if isinstance(v, torch.Tensor):
                    print(k, v.device, v.shape, v.dtype)
                else:
                    print(k, v)

    def print_randn(self):
        print('\n---------------------- create batch -----------------------')
        for i, v in enumerate(self.args):
            if isinstance(v, torch.Tensor):
                print(i, v.device, v.shape, v.dtype)
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                print(i, v[0].device, [a.shape for a in v], [a.dtype for a in v])
            else:
                print(i, v)
        if self.has_kwargs():
            for k, v in self.kwargs.items():
                if isinstance(v, torch.Tensor):
                    print(k, v.device, v.shape, v.dtype)
                elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                    print(k, v[0].device, [a.shape for a in v], [a.dtype for a in v])
                else:
                    print(k, v)

    def create(self):
        global configs
        dtype = self.get_dtype(test_case=self.test_case)
        device = 'cpu' if torch.distributed.is_initialized() else configs['DEVICE']
        if self.by_new_style:
            self.random_data_new(self.test_case.input, dtype, device, self.config)
        else:
            self.random_data(self.test_case.input, dtype, device, self.config)
        if self.test_case.output:
            self.by_list()
        self.scales.clear()
        self.print_randn()

    def clone(self):
        self._args = clone_by_loop_process(self.args)
        self._kwargs = clone_by_loop_process(self.kwargs)
        self._out_info = []
        return self

    def to(self, device, use_cpu_hight_precision=False, fp8_transfer=False):
        self._args = to_device(torch.device(device), self._args, use_cpu_hight_precision, fp8_transfer)
        if self.has_kwargs():
            self._kwargs = to_device(torch.device(device), self._kwargs, use_cpu_hight_precision, fp8_transfer)
        return self

    def has_kwargs(self):
        return self.len_kwargs > 0

    def get_input(self):
        if self.fp8_index:
            for i in self.fp8_index:
                if isinstance(i, int):
                    self._args[i] = self._args[i].to(torch.float32)
                elif isinstance(i, str):
                    self._kwargs[i] = self._kwargs[i].to(torch.float32)
        return self._args, self._kwargs

    def get_tensor_info(self, _tensor, k=None):
        return get_tensor_info_by_loop_process(k, _tensor)

    def get_output(self):
        if self.out_mapping:
            if isinstance(self.out_mapping[0], int):
                result_list = []
                for index in self.out_mapping:
                    item = self._args[index]
                    if isinstance(item, (list, tuple)):
                        # Handle tensor list - move each tensor to CPU
                        result_list.append([tensor.detach().to('cpu') for tensor in item])
                        self._out_info.append(self.get_tensor_info(item))
                    else:
                        # Handle single tensor
                        self._out_info.append(self.get_tensor_info(item))
                        result_list.append(item.detach().to('cpu'))
                return result_list
            else:
                self._out_info.extend([self.get_tensor_info(item, k) for k, item in self._kwargs.items()])
                return [self._kwargs[key].detach().to('cpu') for key in self.out_mapping]

    def check_output(self, output):
        if isinstance(output, (list, tuple)):
            for item in output:
                self._out_info.append(self.get_tensor_info(item))
        elif isinstance(output, dict):
            for k, item in output.items():
                self._out_info.append(self.get_tensor_info(item, k))
        else:
            self._out_info.append(self.get_tensor_info(output))

    def get_case_info(self):
        return {
            'name': self.test_case.name,
            'seed': self.seed,
            'test_case': self.test_case,
            'config': self.config,
            'out': self._out_info
        }