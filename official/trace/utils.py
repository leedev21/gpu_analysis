import os
from typing import List
from typing import Union, Tuple
import torchtrace.optrace.optrace_py as optrace
import torch
import yaml


def current_stream():
    if DEVICE:
        return torch.cuda.current_stream().device_stream
    else:
        return torch.cuda.current_stream().cuda_stream

pid = os.getpid()

OPTRACE_VERSION = 'A'
pyid2idx = {}
idxs = {}

DEFAULT_DATA_PATH = 'data/pt'
GLOBAL_CFG = {
    'sync_mode': False,
    'data_pt_path': DEFAULT_DATA_PATH,
    'state': 'start',
    'step': 0,
    'iter': 0,
    'moduel_save_state': False,
    'c10_by_pass': False,
    'c10_vit_real_tensor': True,
    'nccl_func_save_log': False,
    'comm_init': False,
    'skip_mode': True,
    'print_log': False,
    'nvtx': False,
    'async_save': False,
    'calculate_tensor': 'calc',
    'check_te_tensor': False,
    'nvtx_args': {
        'start': 0,
        'end': 30,
        'ranks': [0,1,2,3,4,5,6,7],
        'layer': ['TransformerLayer', 'DecoderLayer', 'DiTBlock', 'SelfAttention', 'CrossAttention', 'MLP', 'MoE']
        }
}

def get_run_type():
    run_type = os.getenv('RUN_TYPE', '')
    set_failed = False
    if run_type != '':
        run_type_args = run_type.split(';')
        if run_type_args[0] == 'nsys' or run_type_args[0] == 'nsys-shape':
            GLOBAL_CFG['nvtx'] = True
            if len(run_type_args) > 1:
                for item in run_type_args[1:]:
                    if '=' not in item:
                        continue
                    k, v = item.split('=')
                    if k in ['start', 'end', 'ranks']:
                        GLOBAL_CFG['nvtx_args'][k] = eval(v)
                    elif k == 'layer':
                        GLOBAL_CFG['nvtx_args'][k] = v[1:-1].split(',')
                    else:
                        print('Err config, try to set:', k, v)
                        set_failed = True
            print('Enable nsys profile:',  GLOBAL_CFG['nvtx_args'])
        elif run_type_args[0] == 'perf':
            GLOBAL_CFG['perf'] = True
    if set_failed:
        exit(0)

get_run_type()

def load_default(file):
    config = os.path.join(os.path.dirname(__file__) + '/library', file)
    with open(config, 'r') as cfg:
        config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
    return config_dict

def extract_number(text):
    num = ''
    for char in text:
        if char.isdigit():
            num += char
        else:
            break
    return int(num) if num else None

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
TORCH_LAST = extract_number(torch.__version__.split(".")[2])

pt_version = f"{TORCH_MAJOR}_{TORCH_MINOR}_{TORCH_LAST}"

def class_name(obj):
    return obj.__class__.__name__ if obj is not None else ''

def get_obj_id(obj: object):
    py_id = id(obj)
    if py_id not in pyid2idx:
        pyid2idx[py_id] = len(pyid2idx)
    idx = pyid2idx[py_id]
    return idx

def get_saved_tensor_id(idx):
    global idxs
    if idx not in idxs:
        idxs[idx] = 0
        return f'{idx}.{idxs[idx]}', True
    idxs[idx] += 1
    return f'{idx}.{idxs[idx]}', True

def format_tensor_dtype(dtype: torch.dtype):
    # torch.dtype 的取值范围请参考 https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype
    dtype2str = {
        torch.float32 : optrace.datatype.F32,
        torch.float : optrace.datatype.F32,
        torch.float64: optrace.datatype.F64,
        torch.double : optrace.datatype.F64,
        torch.complex64 : optrace.datatype.C64,
        torch.complex128 : optrace.datatype.C128,
        torch.cdouble : optrace.datatype.C128,
        torch.float16 : optrace.datatype.F16,
        torch.half : optrace.datatype.F16,
        torch.bfloat16 : optrace.datatype.BF16,
        torch.uint8 : optrace.datatype.U8,
        torch.int8 : optrace.datatype.I8,
        torch.int16 : optrace.datatype.I16,
        torch.short : optrace.datatype.I16,
        torch.int32 : optrace.datatype.I32,
        torch.int : optrace.datatype.I32,
        torch.int64 : optrace.datatype.I64,
        torch.long : optrace.datatype.I64,
        torch.uint64 : optrace.datatype.U64,
        torch.bool : optrace.datatype.BOOL,
        torch.float8_e4m3fn : optrace.datatype.F8E4M3,
        torch.float8_e5m2 : optrace.datatype.F8E5M2,
    }
    return dtype2str.get(dtype, optrace.datatype.CUSTOM_DATA_TYPE)


def format_obj_name(obj):
    name = class_name(obj)
    obj2str = {
        "float": optrace.datatype.F32,
        "int": optrace.datatype.I32,
    }
    return obj2str.get(name, name)


def format_shape(shape: torch.Size):
    myShape = [i for i in shape]
    return myShape


def format_scaler_dtype(dtype: str):
    # torch.dtype 的取值范围请参考 https://pytorch.org/docs/stable/tensor_attributes.html#torch-dtype
    dtype2str = {
        "float" : optrace.datatype.F32,
        "int" : optrace.datatype.I32,
        "str": optrace.datatype.STR,
    }
    return dtype2str.get(dtype, optrace.datatype.CUSTOM_DATA_TYPE)


def get_device_id(args):
    device_id = None
    if isinstance(args, (list, tuple)):
        for arg in args:
            if isinstance(arg, torch.Tensor):
                device_id = arg.device.index
                break
    else:
        if isinstance(args, torch.Tensor):
            device_id = args.device.index
    return device_id


def read_file(path, file):
    lines = []
    with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
        for item in f.readlines():
            lines.append(item.strip())
    return lines


def print_rank_0(*args):
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    if rank == 0:
        print(*args)

#########################################

def get_type_name(obj):
    # 获取对象的类型
    obj_type = type(obj)

    # 获取类型的字符串表示
    type_str = str(obj_type)

    # 提取类型名称，去掉 <class ' 和 '>
    type_name = type_str.split("'")[1]

    return type_name

parser_map = {}

def register_parser(name: Union[str, List[str], Tuple[str]]):

    def decorator(func):
        if isinstance(name, str):
            parser_map[name] = func
        else:
            for n in name:
                parser_map[n] = func

    return decorator


def calculate_tensor(trace_str, _tensor):
    _min, _max, _bins = -151, 128, 280

    # 计算张量信息用于日志
    if hasattr(_tensor, 'numel'):
        tensor_numel = _tensor.numel()
        tensor_size_mb = tensor_numel * _tensor.element_size() / (1024 * 1024)

        if tensor_size_mb > 2000:
            print(f"[calculate_tensor] Tensor too large ({tensor_size_mb:.2f}MB > 2000MB), skipping calculation for: {trace_str}")
            return None

    # # 计算预估内存消耗用于日志
    # original_size_mb = tensor_size_mb
    # float32_size_mb = tensor_numel * 4 / (1024 * 1024)  # float32 每个元素 4 字节
    # temp_memory_mb = float32_size_mb  # 估算临时内存约等于 float32 tensor 大小
    # histogram_size_mb = (_bins * 2 * 4) / (1024 * 1024)  # 2个直方图，每个280个float32
    # total_estimated_memory_mb = original_size_mb + float32_size_mb + temp_memory_mb + histogram_size_mb

    # # 记录信息
    # if torch.cuda.is_available() and tensor_device.type == 'cuda':
    #     current_memory = torch.cuda.memory_allocated(tensor_device) / (1024 * 1024)
    #     free_memory = torch.cuda.get_device_properties(tensor_device).total_memory / (1024 * 1024) - current_memory
    #     memory_status = "OK" if free_memory > total_estimated_memory_mb else "INSUFFICIENT"
    #     print(f"[calculate_tensor] {trace_str} | {tensor_shape} {tensor_dtype} | Size: {original_size_mb:.1f}MB | Est.Memory: {total_estimated_memory_mb:.1f}MB | GPU: {current_memory:.1f}/{free_memory:.1f}MB | {memory_status}")
    # else:
    #     print(f"[calculate_tensor] {trace_str} | {tensor_shape} {tensor_dtype} | Size: {original_size_mb:.1f}MB | Est.Memory: {total_estimated_memory_mb:.1f}MB | Device: {tensor_device}")

    positive = torch.histc(torch.log2(_tensor.to(torch.float)), min=_min, max=_max, bins=_bins)
    negative = torch.histc(torch.log2(-_tensor.to(torch.float)), min=_min, max=_max, bins=_bins)
    return {'name': trace_str, 'd': torch.stack((positive, negative), dim=0)}

def calculate_tensor_to_list(_tensor):
    _min, _max, _bins = -151, 128, 280
    positive = torch.histc(torch.log2(_tensor.to(torch.float)), min=_min, max=_max, bins=_bins)
    negative = torch.histc(torch.log2(-_tensor.to(torch.float)), min=_min, max=_max, bins=_bins)
    return positive.tolist(), negative.tolist()

def calculate_tensor_to_dict(_tensor):
    positive, negative = calculate_tensor_to_list(_tensor)
    res = {}
    for i in range(len(positive)):
        if positive[i] > 0:
            res[f"e{i-151}"] = positive[i]
    for i in range(len(negative)):
        if negative[i] > 0:
            res[f"-e{i-151}"] = negative[i]
    return res

import torch.distributed.checkpoint as dcp

def _get_save_mode(trace_type):
    try:
        default_cfg = GLOBAL_CFG[trace_type]
        GLOBAL_CFG['calculate_tensor'] = default_cfg.get('save_tensor', 'calc')
    except:
        GLOBAL_CFG['calculate_tensor'] = 'calc'
    return GLOBAL_CFG['calculate_tensor']

def _check_tensor_anomalies(tensor, trace_str):
    """检测tensor异常值"""
    try:
        if not isinstance(tensor, torch.Tensor):
            return

        # 从trace_str中提取算子名
        op_name = trace_str.split(':')[-1] if ':' in trace_str else trace_str
        anomalies = []

        # 1. 检测特殊值（NaN、Inf） - 只对浮点数tensor
        if tensor.dtype.is_floating_point:
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            if has_nan:
                anomalies.append("Contains NaN values")
            if has_inf:
                anomalies.append("Contains Inf values")

        # 2. 检测零tensor
        if tensor.numel() > 0:
            all_zero = (tensor == 0).all().item()
            if all_zero:
                anomalies.append("All values are zero")

        # 3. 检测空tensor
        if tensor.numel() == 0:
            anomalies.append("Empty tensor")

        # 4. 检测异常大的数值（只对浮点数）
        if tensor.dtype.is_floating_point and tensor.numel() > 0:
            abs_max = tensor.abs().max().item()
            if abs_max > 1e6:  # 大于100万
                anomalies.append(f"Very large values (max: {abs_max:.2e})")
            elif abs_max > 0 and abs_max < 1e-6:  # 非零但小于1e-6
                anomalies.append(f"Very small values (max: {abs_max:.2e})")

        # 5. 检测梯度爆炸/消失（如果tensor需要梯度）
        if hasattr(tensor, 'grad') and tensor.grad is not None:
            grad_norm = tensor.grad.norm().item()
            if grad_norm > 100:
                anomalies.append(f"Large gradient norm: {grad_norm:.2e}")
            elif grad_norm < 1e-8:
                anomalies.append(f"Very small gradient norm: {grad_norm:.2e}")

        # 6. 数据分布检查：方差过小/过大
        if tensor.dtype.is_floating_point and tensor.numel() > 1:
            try:
                tensor_var = tensor.var().item()
                if tensor_var > 1e3:  # 方差过大
                    anomalies.append(f"High variance: {tensor_var:.2e}")
                elif tensor_var < 1e-8 and tensor_var > 0:  # 方差过小但非零
                    anomalies.append(f"Very low variance: {tensor_var:.2e}")
            except:
                pass

        # 7. 稀疏性检查：零值比例过高
        if tensor.numel() > 0:
            zero_ratio = (tensor == 0).float().mean().item()
            if zero_ratio > 0.9:  # 超过90%是零值
                anomalies.append(f"High sparsity: {zero_ratio*100:.1f}% zeros")

        # 8. 数值稳定性：条件数检查（只对2D矩阵）
        if tensor.dtype.is_floating_point and len(tensor.shape) == 2 and min(tensor.shape) > 1:
            try:
                # 计算条件数（奇异值的最大值/最小值）
                U, S, V = torch.svd(tensor.float())
                if S.numel() > 0:
                    max_sv = S.max().item()
                    min_sv = S.min().item()
                    if min_sv > 1e-12:  # 避免除零
                        cond_num = max_sv / min_sv
                        if cond_num > 1e12:  # 条件数过大，数值不稳定
                            anomalies.append(f"Poor numerical stability (cond: {cond_num:.2e})")
                        elif min_sv < 1e-10:  # 最小奇异值过小，接近奇异
                            anomalies.append(f"Near singular matrix (min_sv: {min_sv:.2e})")
            except:
                pass  # SVD可能失败，忽略错误

        # 只有发现异常才打印
        if anomalies:
            print(f"[TENSOR ANOMALY] {op_name}")
            for anomaly in anomalies:
                print(f"{anomaly}")
            print(f"Shape: {list(tensor.shape)}, DType: {tensor.dtype}")
            if tensor.device.type != 'cpu':
                print(f"Device: {tensor.device}")

    except Exception as e:
        print(f"[TensorAnalysis] Error: {e}")

def _prepare_tensor_for_save(trace_type, t: torch.Tensor, trace_str: str):
    save_mode = _get_save_mode(trace_type)

    if save_mode == 'tensor':
        # 将tensor转移到CPU设备上进行保存
        tensor_cpu = t.cpu()
        return True, {'name': trace_str, 'd': tensor_cpu}, None
    elif save_mode == 'calc':
        calc_data = calculate_tensor(trace_str, t)
        return True, None, calc_data
    elif save_mode == 'all':
        # 将tensor转移到CPU设备上进行保存
        tensor_cpu = t.cpu()
        tensor_data = {'name': trace_str, 'd': tensor_cpu}
        calc_data = calculate_tensor(trace_str, t)
        return True, tensor_data, calc_data
    elif save_mode == 'analysis':
        # analysis模式：不保存tensor数据，只进行实时分析
        _check_tensor_anomalies(t, trace_str)
        return False, None, None

    return False, None, None

def _get_tensor_save_path(trace_type, save_prefix: str, idx: str):
    if trace_type == "tensor":
        filename = f'tensor_{save_prefix}_{idx}.pt'
    else:
        filename = f'{save_prefix}_{idx}.pt'

    return os.path.join(get_save_path(), filename)

def _move_tensors_to_cpu(data):
    """递归地将数据中的所有tensor移动到CPU设备上"""
    if isinstance(data, torch.Tensor):
        return data.cpu(), str(data.size())
    elif isinstance(data, dict):
        return {k: _move_tensors_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_move_tensors_to_cpu(item) for item in data)
    else:
        return data, str([0])

def _save_tensor_data(save_path, data):
    try:
        # 确保数据中的所有tensor都在CPU上
        cpu_data, data_shape = _move_tensors_to_cpu(data)
        with open(os.path.join(get_save_path(), 'op_dump.txt'), 'a+', encoding='utf-8') as f:
            f.write(save_path + ' ' + str(data_shape))
        if GLOBAL_CFG['async_save']:
            dcp.async_save(cpu_data, checkpoint_id=save_path.replace('.pt', ''))
        else:
            torch.save(cpu_data, save_path)
    except Exception as e:
        print(f"[Error] Failed to save tensor data to {save_path}: {e}")


def _tensor_trace(idx, t, dtype, size, stride, storage_offset, save):
    trace = optrace.tensor(idx, format_tensor_dtype(dtype), format_shape(size), stride, storage_offset)
    trace_str = trace.getString().split(':')[1]
    if save:
        idx_save, new = get_saved_tensor_id(idx)
        if not new:
            idx_save = ''
        if hasattr(t, 'numel'):
            tensor_numel = t.numel()
            tensor_size_mb = tensor_numel * t.element_size() / (1024 * 1024)
            if tensor_size_mb > 1024:
                print(f"[tensor_parser] Tensor too large ({tensor_size_mb:.2f}MB > 1024MB), skipping calculation for: {trace_str}")
                idx_save = ''
    else:
        idx_save = ''
    if idx_save:
        trace_str = f'%{idx_save}:{trace_str}'
        should_save_tensor, tensor_data, calc_data = _prepare_tensor_for_save('op_trace', t, trace_str)
        if should_save_tensor:
            if tensor_data:
                _save_tensor_data(_get_tensor_save_path('tensor', save, idx_save), tensor_data)
            if calc_data:
                _save_tensor_data(_get_tensor_save_path('calc', save, idx_save), calc_data)
    return trace, trace_str


@register_parser(["Float8Tensor", "Float8TensorBase", "MXFP8Tensor", "MXFP8TensorBase", "Float8BlockwiseQTensor", "Float8BlockwiseQTensorBase"])
def tensor_parser(t: torch.Tensor, save=False):
    from transformer_engine.pytorch.constants import TE_DType_To_Torch
    dtype = TE_DType_To_Torch[t._fp8_dtype] if hasattr(t, '_fp8_dtype') else t.dtype
    if hasattr(t, '_columnwise_scale_inv') and hasattr(t, '_columnwise_data') and t._columnwise_data is not None:
        value = '2D' if t._is_2D_scaled else '1D'
        type_trace = optrace.scalar(optrace.datatype.CUSTOM_DATA_TYPE, value, 'Mode')
        data_format = optrace.scalar(optrace.datatype.CUSTOM_DATA_TYPE, t._data_format.name, 'Format')
        idx1 = get_obj_id(t._columnwise_data)
        tensor_1, tensor_1_str = _tensor_trace(idx1, t._columnwise_data, t._columnwise_data.dtype, t._columnwise_data.size(), [], 0, save)
        scale_idx1 = get_obj_id(t._columnwise_scale_inv)
        scale_1, scale_1_str = _tensor_trace(scale_idx1, t._columnwise_scale_inv, t._columnwise_scale_inv.dtype, t._columnwise_scale_inv.size(), [], 0, save)
        if hasattr(t, '_rowwise_data') and t._rowwise_data is not None:
            idx2 = get_obj_id(t._rowwise_data)
            tensor_2, tensor_2_str = _tensor_trace(idx2, t._rowwise_data, t._rowwise_data.dtype, t._rowwise_data.size(), [], 0, save)
            scale_idx2 = get_obj_id(t._rowwise_scale_inv)
            scale_2, scale_2_str = _tensor_trace(scale_idx2, t._rowwise_scale_inv, t._rowwise_scale_inv.dtype, t._rowwise_scale_inv.size(), [], 0, save)
            trace = optrace.structure(idx1, "tuple", [tensor_1, scale_1, tensor_2, scale_2, type_trace, data_format])
            trace_str = "tuple{" + ', '.join([tensor_1_str, scale_1_str, tensor_2_str, scale_2_str, type_trace.getString(), data_format.getString()]) + "}"
        else:
            trace = optrace.structure(idx1, "tuple", [tensor_1, scale_1, type_trace, data_format])
            trace_str = "tuple{" + ', '.join([tensor_1_str, scale_1_str, type_trace.getString(), data_format.getString()]) + "}"
    elif hasattr(t, '_rowwise_scale_inv'):
        value = '2D' if t._is_2D_scaled else '1D'
        type_trace = optrace.scalar(optrace.datatype.CUSTOM_DATA_TYPE, value, 'Mode')
        data_format = optrace.scalar(optrace.datatype.CUSTOM_DATA_TYPE, t._data_format.name, 'Format')
        idx2 = get_obj_id(t._rowwise_data)
        tensor_2, tensor_2_str = _tensor_trace(idx2, t._rowwise_data, t._rowwise_data.dtype, t._rowwise_data.size(), [], 0, save)
        scale_idx2 = get_obj_id(t._rowwise_scale_inv)
        scale_2, scale_2_str = _tensor_trace(scale_idx2, t._rowwise_scale_inv, t._rowwise_scale_inv.dtype, t._rowwise_scale_inv.size(), [], 0, save)
        trace = optrace.structure(idx2, "tuple", [tensor_2, scale_2, type_trace, data_format])
        trace_str = "tuple{" + ', '.join([tensor_2_str, scale_2_str, type_trace.getString(), data_format.getString()]) + "}"
    else:
        idx = get_obj_id(t)
        trace, trace_str = _tensor_trace(idx, t, dtype, t.size(), [], 0, save)
        if hasattr(t, '_scale_inv'):
            scale = t._scale_inv
            scale_idx = get_obj_id(scale)
            dtype = scale.dtype if hasattr(scale, 'dtype') else None
            scale_trace, scale_str = _tensor_trace(scale_idx, scale, dtype, scale.size(), [], 0, save)
            trace = optrace.structure(idx, "tuple", [trace, scale_trace])
            trace_str = "tuple{" + ', '.join([trace_str, scale_str]) + "}"

    return trace, trace_str


@register_parser(["Tensor", "DTensor", "Parameter", "FunctionalTensor", "FakeTensor", "AsyncCollectiveTensor"])
def tensor_parser(t: torch.Tensor, save=False):
    idx = get_obj_id(t)
    if len(t.size()) > 0:
        #create tensor: tensorID + data type + dims + strides + offset
        dtype = t.dtype if hasattr(t, 'dtype') else None
        stride = t.stride() if hasattr(t, 'stride') else []
        storage_offset = t.storage_offset() if hasattr(t, 'storage_offset') else 0
        return _tensor_trace(idx, t, dtype, t.size(), stride, storage_offset, save)
    else:
        # Represent 0-d tensors as a 1-element tensor to keep shape formatting
        trace_type = format_tensor_dtype(t.dtype) if hasattr(t, 'dtype') else optrace.datatype.CUSTOM_DATA_TYPE
        try:
            trace = optrace.tensor(idx, trace_type, [1], [1], 0)
        except Exception:
            trace = optrace.scalar(idx, trace_type, str(t.item()))
        trace_str = trace.getString().split(':')[1]
    return trace, trace_str


@register_parser(["UnpackedDualTensor"])
def tensor_parser(var: torch.autograd.forward_ad.UnpackedDualTensor, save=False):
    t = var.primal
    idx = get_obj_id(t)
    return _tensor_trace(idx, t, t.dtype, t.shape, t.stride(), t.storage_offset(), save)


@register_parser(["list"])
def list_parser(t: list, save=False):
    idx = get_obj_id(t)
    data = []
    save_res = []
    for i, elem in enumerate(t):
        parser = parser_map.get(class_name(elem),
                                            default_parser)
        trace_elem, saved = parser(elem, save)
        data.append(trace_elem)
        save_res.append(saved)
    #create structure: id + name + data
    trace = optrace.structure(idx, "list", data)
    #trace = optrace.structure("list", data)            #TODO:需要支持不带idx的构造函数
    if save:
        return trace, save_res
    return trace, "list{" + ', '.join(save_res) + "}"


@register_parser(["tuple"])
def tuple_parser(t: tuple, save=False):
    idx = get_obj_id(t)
    data = []
    save_res = []
    for i, elem in enumerate(t):
        parser = parser_map.get(class_name(elem),
                                            default_parser)
        trace_elem, saved = parser(elem, save)
        data.append(trace_elem)
        save_res.append(saved)
    #create structure: id + name + data
    trace = optrace.structure(idx, "tuple", data)
    #trace = optrace.structure("tuple", data)         #TODO:需要支持不带idx的构造函数
    if save:
        return trace, save_res
    return trace, "tuple{" + ', '.join(save_res) + "}"


#@register_parser("UntypedStorage") #TODO，需要修改这个函数用来支持特定类型的trace，并增加测试用例
#def untypedstorage_parser(info: str, var):
#    info += f"storage{{size={var.nbytes()}}}"
#    return info

@register_parser(["ProcessGroup", "ReduceOp", "Work", "ScriptObject", "Float8BlockQuantizer",
                 "Float8Quantizer", "Float8CurrentScalingQuantizer", "MXFP8Quantizer", "_Float8BlockQuantizer",
                 "_Float8Quantizer", "_Float8CurrentScalingQuantizer", "_MXFP8Quantizer",])
def tensor_parser(t: object, save=False):
    trace_type = optrace.datatype.CUSTOM_DATA_TYPE
    custom_data_type = class_name(t)
    if custom_data_type == "ProcessGroup":
        value = f"<rank:{t.rank()}, size:{t.size()}, group_name:{t.group_name}, backend:{t._get_backend_name()}>"
    elif custom_data_type == "ReduceOp":
        value = f"{t.__name__}"
    elif custom_data_type == "Work":
        value = f"Work"
    elif custom_data_type == "ScriptObject":
        custom_data_type = 'distributed'
        if 'ProcessGroup' in str(t):
            value = 'ProcessGroup'
        elif "ReduceOp" in str(t):
            value = 'ReduceOp'
        else:
            value = f"Work"
    else:
        value = f"<{t.rowwise_usage}, {t.columnwise_usage}, {t.internal}>"
    trace = optrace.scalar(trace_type, value, custom_data_type)
    return trace, trace.getString()



@register_parser("Generator") # TODO，optrace解析需要支持这种类型吗？
def generator_parser(var, save=False):
    seed = var.initial_seed()
    name = "seed"
    trace_type = format_obj_name(seed)
    value = str(seed)
    custom_data_type = get_type_name(seed)
    trace_seed = optrace.scalar(name, trace_type, value, custom_data_type)

    offset = None
    if var.device == 'cuda':
        offset = var.get_offset()
    name = "offset"
    trace_type = optrace.datatype.CUSTOM_DATA_TYPE
    value = str(offset)
    custom_data_type = get_type_name(offset)
    trace_offset = optrace.scalar(name, trace_type, value, custom_data_type)

    data = [trace_seed, trace_offset]
    idx = get_obj_id(var)
    trace = optrace.structure(idx, "tuple", data)
    if save:
        return trace, var
    return trace, trace.getString()


@register_parser("record_function")
def default_parser(t: object, save=False):
    trace_type = optrace.datatype.CUSTOM_DATA_TYPE
    try:
        value = t.name
    except:
        value = "NotSurpot"
    # 如果value中包含关键字，会影响optrace解析。此时需要特殊处理。
    keywords = [":", " ", "<", ">", "{", "}", ".", "x"]
    substitute = "_"
    for sub in keywords:
        if sub in value:
            value = value.replace(sub, substitute)
    #custom_data_type = get_type_name(t) # 使用全称
    custom_data_type = class_name(t) # 不使用全称。
    trace = optrace.scalar(trace_type, value, custom_data_type)
    if save:
        return trace, t
    return trace, f'autograd.profiler.record_function({value})'


@register_parser("bind_default_args")
def default_parser(t: object, save=False):
    trace_type = optrace.datatype.CUSTOM_DATA_TYPE
    try:
        value = t.__dict__
    except:
        value = "NotSurpot"
    # 如果value中包含关键字，会影响optrace解析。此时需要特殊处理。
    keywords = [":", " ", "<", ">", "{", "}", ".", "x"]
    substitute = "_"
    for sub in keywords:
        if sub in value:
            value = value.replace(sub, substitute)
    #custom_data_type = get_type_name(t) # 使用全称
    custom_data_type = class_name(t) # 不使用全称。
    trace = optrace.scalar(trace_type, value, custom_data_type)
    if save:
        return trace, t
    return trace, f'Function.apply.bind_default_args({value})'


def default_parser(t: object, save=False):
    #trace = optrace.scalar(idx, optrace.datatype.I32, f"type(t)={type(t)}, t={t}") 
    #trace = optrace.scalar("customertype", optrace.datatype.CUSTOM_DATA_TYPE, f"t={t}", f"type(t)={type(t)}")
    trace_type = optrace.datatype.CUSTOM_DATA_TYPE
    try:
        if 'torch.autograd.function' in str(t):
            value = t.__class__.__name__
        else:
            value = str(t)
        if "\n" in value:
            value = "NotSurpot"
    except:
        value = "NotSurpot"

    # 如果value中包含关键字，会影响optrace解析。此时需要特殊处理。
    keywords = [":", " ", "<", ">", "{", "}", ".", "x"]
    substitute = "_"
    for sub in keywords:
        if sub in value:
            value = value.replace(sub, substitute)
    #custom_data_type = get_type_name(t) # 使用全称
    custom_data_type = 'autograd' if 'torch.autograd.function' in str(t) else class_name(t) # 不使用全称。

    trace = optrace.scalar(trace_type, value, custom_data_type)

    if save:
        return trace, t
    try:
        return trace, trace.getString()
    except:
        return trace, str(t)


def get_save_path() -> str:
    """
    Get the save path for the trace data.
    The save path is the same as the data_pt_path, but with the rank appended if the world size is greater than 1.
    
    Note: This method must be called after torch.distributed.init_process_group() to correctly get the rank.
    If called before initialization, it will fall back to environment variables RANK and WORLD_SIZE.
    """
    save_to = GLOBAL_CFG['data_pt_path']
    try:
        import torch
        import torch.distributed
        rank = str(torch.distributed.get_rank())
        world_size = torch.distributed.get_world_size()
    except Exception as e:
        print(f"[TorchTrace] Error getting rank and world size: {e}")
        rank = os.getenv('RANK', '-1')
        world_size = int(os.getenv('WORLD_SIZE', '-1'))
    if rank != "-1" and world_size > 1:
        save_to += rank
    if not os.path.isdir(save_to):
        os.makedirs(save_to, exist_ok=True)
    return save_to