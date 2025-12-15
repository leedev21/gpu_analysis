import torch
from .customer import get_op

class OPSwapper():
    def __init__(self, test_case, config):
        self.test_case = test_case
        self.__name__ = 'OPSwapper'

    def swap_input(self, batch, mapping):
        _in = []
        for i, v in mapping:
            if v == 'T':
                _tensor = batch[i].T
            elif v == 'dtype':
                _tensor = batch[i].dtype
            elif v == 'rank':
                _tensor = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            else:
                _tensor = batch[i]
            _in.append(_tensor)
        return _in

    def __call__(self, *batch):
        if batch[0].device.type == 'cpu' and 'cpu_name' in self.test_case:
            print('cpu', self.test_case['cpu_name'])
            package, op_name = self.test_case['cpu_name'].split('::')
            if 'cpu_mapping' in self.test_case:
                batch = self.swap_input(batch, self.test_case['cpu_mapping'])
            if 'output' in self.test_case and self.test_case['output']:
                output = self.test_case['output']
        elif batch[0].device.type == 'cuda' and 'cuda_name' in self.test_case:
            print('cuda', self.test_case['cuda_name'])
            package, op_name = self.test_case['cuda_name'].split('::')
            if 'cuda_mapping' in self.test_case:
                batch = self.swap_input(batch, self.test_case['cuda_mapping'])
            if 'output' in self.test_case and self.test_case['output']:
                output = self.test_case['output']
        else:
            package, op_name = self.test_case.name.split('::')
            output = None
        Op = get_op(package, op_name)
        res = Op(*batch)
        if isinstance(res, torch.Tensor):
            return [res]
        return res

