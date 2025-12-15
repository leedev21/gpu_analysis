import os
import torch
import time

import torch.distributed
from .compare import Comparator, CompareColumn
from .npy_compare import compare_ops_apply, get_error_message
from .advanced_compare import draw_manager


class STATE():
    data_pt_path = 'data/pt'
    device = torch.device('cuda')
    target_device = torch.device('cpu')
    use_cpu_float = True

    def set_pt_path(self, path):
        self.data_pt_path = path
        if not os.path.isdir(self.data_pt_path):
            os.makedirs(self.data_pt_path, exist_ok=True)

RUN_CFG = STATE()


def loop_process_data(device, data, func, exception=None):
    if isinstance(data, dict):
        return type(data)(**{k: loop_process_data(device, v, func) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(loop_process_data(device, v, func) for v in data)
    elif isinstance(data, str) and data.startswith('%'):
        return func(device, data)
    elif isinstance(data, (torch.Tensor, torch.nn.Parameter)):
        return func(device, data.data)
    else:
        try:
            _dict = dict(data)
            return dict(**{k: loop_process_data(device, _dict[k], func) for k in _dict})
        except:
            # print('Err process:'.__class__.__name__, data)
            if exception:
                return exception(device, data)
            else:
                return data


def load_to_device(device, data):
    def load_tensor(device, data):
        file_name = data[1:].split(':')[0] + '.pt'
        _tensor = torch.load(os.path.join(RUN_CFG.data_pt_path, file_name), map_location=device)
        return _tensor['d']
    return loop_process_data(device, data, load_tensor)


def to_device(device, data, use_cpu_hight_precision=False, fp8_transfer=False):
    def tensor_to(device, data):
        if hasattr(data, 'dtype') and device == torch.device('cpu'):
            if use_cpu_hight_precision:
                if data.dtype in [torch.bfloat16, torch.half]:
                    return data.float().to(device)
                if data.dtype in [torch.float32]:
                    return data.double().to(device)
            if fp8_transfer:
                if data.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
                    return data.float().to(device)
        return data.to(device)
    return loop_process_data(device, data, tensor_to)


current_time = time.strftime("%Y%m%d%H%M%S")
RESULT_FILE_NAME = "accuracy_checking_result_" + current_time + ".csv"
DETAILS_FILE_NAME = "accuracy_checking_details_" + current_time + '.csv'
result_csv_path = os.path.join('data/accuracy_check', RESULT_FILE_NAME)
details_csv_path = os.path.join('data/accuracy_check', DETAILS_FILE_NAME)
is_continue_run_ut = False
compare = Comparator(result_csv_path, details_csv_path, is_continue_run_ut)

precision_configs = {
    torch.float16 : {
        'rtol' :  0.5e-3,
        'atol' :  1e-5
    },
    torch.bfloat16: {
        'rtol' : 3.9e-3,
        'atol' : 1e-5
    },
    torch.float32:{
        'rtol' : 1e-6,
        'atol' : 1e-9
    }
}

def get_dtype_to_str(dtype):
    precision_map = {torch.float: 'fp32',
                     torch.half: 'fp16',
                     torch.bfloat16: 'bf16',
                     torch.float8_e4m3fn: 'e4m3',
                     torch.float8_e5m2: 'e5m2',
                     torch.float64: 'fp64',
                     torch.int64: 'long',
                     torch.int32: 'int',
                     torch.int: 'int',
                     torch.bool: 'bool'}
    return precision_map[dtype] if dtype in precision_map else dtype

def check_diff(a, b, backward=False):
    res = []
    if isinstance(a, tuple) or isinstance(a, list):
        for aa, bb in zip(a, b):
            if aa is None or bb is None:
                res.append("None")
                continue
            res.extend(check_diff(aa, bb, backward))
    elif isinstance(a, torch.Tensor):
        dtype = a.dtype
        limit = precision_configs[dtype] if dtype in precision_configs else precision_configs[torch.float32]
        if not res:
            # res.append(dtype)
            res.append(get_dtype_to_str(b.dtype))
            res.append(limit['rtol'])
            res.append(limit['atol'])
            res.append(get_dtype_to_str(a.dtype))
        # res.append(a.shape)
        a_cpu = a.cpu()
        b_cpu = b.cpu()
        if not backward:
            check_pass = torch.allclose(a_cpu.float(), b_cpu.float(),
                                        rtol=limit['rtol'],
                                        atol=limit['atol'])
        else:
            check_pass = torch.allclose(a_cpu.grad.float(), b_cpu.grad.float(),
                                        rtol=limit['rtol'],
                                        atol=limit['atol'])
        res.insert(0, check_pass)
        # if not check_pass:
        #     print(a_cpu.shape, b_cpu.shape, check_diff_one(a_cpu, b_cpu))
    return res


def check_diff_one(a, b):
    line = str(b)[:50] + ' ,..., ' + str(b)[-50:]
    if isinstance(a, torch.Tensor):
        loss = torch.abs(b.float() - a.float())
        _avg = torch.mean(loss).cpu().tolist()
        _var = torch.var(loss).cpu().tolist()
        _max = torch.max(loss).cpu().tolist()
        return line, _avg, _var, _max
    else:
        return None, None, None, None


def run_op(Op, device, inputs):
    args, kwargs = inputs
    if hasattr(Op, 'to'):
        Op = Op.to(device)
    return Op(*args, **kwargs)


def loop_print(data):
    if isinstance(data, (tuple, list)):
        for a in data:
            loop_print(a)
    else:
        print(data)


def each_file(dir, endwith='.pt', with_path=False):
    paths = os.walk(dir)
    for path, dir_lst, file_lst in paths:
        for file_name in file_lst:
            if file_name.endswith(endwith):
                if with_path:
                    yield path, os.path.join(path, file_name)
                else:
                    yield os.path.join(path, file_name)


def acc_check(api_name, device, target, out, use_allclose, use_detail_check, grad=False):
    if use_allclose:
        detail_msg = []
        check_msg = check_diff([out], [target])
        n_out_tensor = len(check_msg) / 5
        for i in range(int(n_out_tensor)):
            _msg = check_msg[i*5:(i+1)*5]
            _msg[0] = 'pass' if _msg[0] else 'error'
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
            draw_manager.set_title(api_name, rank, device, i, _msg[1], _msg[-1])
            if isinstance(out, (list, tuple)):
                _target, _out = target[i], out[i]
            else:
                _target, _out = target, out
            _msg.insert(1, device)
            _msg.pop(-1)
            if True:                  # if _msg[0] != 'pass': need to run for drawing
                # Check if target and out are valid tensors
                if _target is None or _out is None:
                    print('target is None or out is None')
                    _msg += ["error", "target or out is None", "", ""]
                    continue
                elif not isinstance(_target, torch.Tensor) or not isinstance(_out, torch.Tensor):
                    if isinstance(_target, tuple) and isinstance(_target[1], torch.Tensor) \
                            and isinstance(_out, tuple) and isinstance(_out[1], torch.Tensor):
                        _target = _target[1]
                        _out = _out[1]
                    else:
                        print(f'target type: {type(_target)}, out type: {type(_out)}')
                        _msg += ["error", f"target type: {type(_target)}, out type: {type(_out)}", "", ""]
                        continue
                if not use_detail_check:
                    _n = _target.float().cpu().numpy()
                    _b = _out.float().cpu().numpy()
                    err_msg = get_error_message(_n, _b, api_name, False, error_file=None)
                    err_msg, _ = compare_ops_apply(_n, _b, False, err_msg, relative_err=None)
                    _msg += err_msg
                else:
                    status, err_msg = compare._compare_core_wrapper(api_name, _target, _out, grad=grad)
                    _msg.extend(err_msg[0])
                    if _msg[0] == 'pass':
                        if status != 'pass':
                            _msg[0] = 'Warning'
                    else:
                        _msg[0] = status
            detail_msg.append(_msg)
    else:
        if not use_detail_check:
            if isinstance(out, tuple):
                target = target[0] if target is not None else None
                out = out[0] if out is not None else None
            # Check if target and out are valid tensors
            if target is None or out is None:
                detail_msg = ["error", "target or out is None", "", ""]
                return detail_msg
            elif not isinstance(target, torch.Tensor) or not isinstance(out, torch.Tensor):
                detail_msg = ["error", f"target type: {type(target)}, out type: {type(out)}", "", ""]
                return detail_msg
            else:
                _n = target.float().cpu().numpy()
                _b = out.float().cpu().numpy()
                detail_msg = get_error_message(_n, _b, api_name, False, error_file=None)
                status, detail_msg = compare_ops_apply(_n, _b, False, detail_msg, relative_err=None)
        else:
            status, detail_msg = compare._compare_core_wrapper(api_name, target, out, grad=grad)
    return detail_msg


class AccCheckHelper(object):
    def __init__(self):
        self.data = []

    def add(self, _i, model, test_case, hw_name, tensor):
        api_name = model.__name__
        self.data.append({'test_case': test_case, 'api_name': api_name, 'device': 'cpu', 'tensor': tensor})

    def check(self, args):
        res = []
        target = None
        dynamic_target = len(self.data) > 10
        for i, data in enumerate(self.data):
            if i == 0:
                target = data['tensor']
            else:
                res.extend(acc_check(data['api_name'], data['device'], data['tensor'], target, args.run.use_allclose, args.run.use_detail_check))
                if dynamic_target:
                    target = data['tensor']
        return res

    def check_this(self, args):
        res = self.check(args)
        if len(self.data) > 1:
            del self.data[1]
        return res

    def clear(self):
        self.data.clear()


acc_check_helper = AccCheckHelper()