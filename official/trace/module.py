import os
import torch
import time
import datetime
import json
import yaml
import glob
from torchtrace.utils import (
    OPTRACE_VERSION,
    pid,
    get_device_id,
    pt_version,
    current_stream,
    parser_map,
    class_name,
    default_parser,
    read_file,
    GLOBAL_CFG,
    get_save_path
)
from torchtrace.library.kv_cache import customer_kv_cache, prepare_kv, prepare_for_send
from torchtrace.library.fallback_gpu import GPUService


class GPUServiceConfig:
    host = '172.0.0.1'
    port = 1234
    nfs_path = ''
    tls_path = ''
    # rank = [0, 1, 2, 3, 4, 5, 6, 7]
    rank = [-1]
    is_online = True
# class GPUServiceConfig:
#     host = '10.9.112.129'
#     port = 49624
#     nfs_path = ''
#     tls_path = ''
#     # rank = [0, 1, 2, 3, 4, 5, 6, 7]
#     rank = []
#     is_online = True


class Tracker():
    def __init__(self):
        self.state = 'start'
        self.step = 0
        self.iter = 0
        self.model = {}
        self.module = {}
        self.ops = {}
        self.op_info = {}
        self.comms = {}
        self.apis = {}
        self.npz_num = 0
        self.state_stack = []
        self.api_stack = []
        self.schemas = {}
        self.fw_version = {}
        self.rank = int(os.environ.get('LOCAL_RANK', 0))
        self.support_by_torch = {}
        for line in read_file(os.path.dirname(__file__) + '/library', 'support_by_torch.txt'):
            op, no_need_skip = line.split(' ')
            self.support_by_torch[op] = eval(no_need_skip)
        # 新增：为目标格式准备的数据结构
        self.ops_detailed = {}  # 存储详细的操作信息
        self.op_schemas = {}    # 存储操作的schema信息
        self.op_aten_map = {}  # 存储操作到aten的映射
        self.fallback_service = GPUService(GPUServiceConfig)

    def _find_pt_file(self, aten_op_name, tensor_id, name, is_output=False):
        """查找对应的pt文件路径"""
        base_save_path = get_save_path()
        if not base_save_path or not os.path.exists(base_save_path):
            return None

        # 构建搜索模式
        if is_output:
            pattern = f"*tensor*{aten_op_name}_out_{tensor_id}.pt"
        else:
            pattern = f"*tensor*{aten_op_name}_{name}_{tensor_id}.pt"

        # 在指定路径下搜索文件
        search_path = os.path.join(base_save_path, pattern)
        matching_files = glob.glob(search_path)

        if matching_files:
            # 如果找到多个文件，返回第一个，并去掉vllm_config_path的路径
            full_path = matching_files[0]
            relative_path = os.path.relpath(full_path, base_save_path)
            return relative_path

        return None

    def _parse_tensor_info(self, tensor_str, aten_op_name=None, name=None, is_output=False):
        """解析tensor字符串，提取详细信息"""
        import re

        # 处理两种格式：
        # 1. %107.0:<1024x16x80xbf16>{1280, 80, 1}
        # 2. <8x151936xf32>{151936, 1}

        tensor_id = None

        # 先尝试匹配完整格式
        pattern1 = r'%(\d+\.\d+):<(.+?)>(\{.+?\})'
        match1 = re.search(pattern1, tensor_str)

        if match1:
            tensor_id = match1.group(1)
            size_dtype_str = match1.group(2)
            stride_str = match1.group(3)
        else:
            # 尝试匹配简化格式
            pattern2 = r'<(.+?)>(\{.+?\})'
            match2 = re.search(pattern2, tensor_str)
            if match2:
                size_dtype_str = match2.group(1)
                stride_str = match2.group(2)
            else:
                return None

        # 解析size和dtype
        size, dtype_str = self._parse_size_dtype(size_dtype_str)

        # 解析stride
        stride_match = re.search(r'\{(.+?)\}', stride_str)
        strides = []
        if stride_match:
            stride_content = stride_match.group(1)
            if stride_content.strip():
                strides = [int(x.strip()) for x in stride_content.split(',') if x.strip()]

        # 映射dtype
        dtype_map = {
            'bf16': 'BFloat16',
            'bfloat16': 'BFloat16',
            'f32': 'Float',
            'float32': 'Float',
            'float': 'Float',
            'f16': 'Half',
            'float16': 'Half',
            'half': 'Half',
            'i32': 'Int',
            'int32': 'Int',
            'i64': 'Long',
            'int64': 'Long',
            'long': 'Long',
            'bool': 'Bool',
            'u8': 'Byte',
            'uint8': 'Byte',
            'i8': 'Char',
            'int8': 'Char',
            'f8e4m3': 'Float8E4M3FN',
            'f8e5m2': 'Float8E5M2',
            'float8_e4m3fn': 'Float8E4M3FN',
            'float8_e5m2': 'Float8E5M2',
        }

        dtype = dtype_map.get(dtype_str, 'Float')

        # 计算is_contiguous
        is_contiguous = self._check_contiguous(size, strides)

        # 构建基本的tensor信息
        tensor_info = {
            "dtype": dtype,
            "is_contiguous": is_contiguous,
            "size": size,
            "strides": strides
        }

        # 如果启用了save_pt并且有tensor_id，尝试查找对应的pt文件
        if GLOBAL_CFG.get('save_pt', False) and tensor_id and aten_op_name and name:
            pt_path = self._find_pt_file(aten_op_name, tensor_id, name, is_output)
            tensor_info["tensor"] = pt_path
            # 新增distribution字段，去掉pt_path开头的"tensor"
            if pt_path and pt_path.startswith("tensor"):
                # 去掉"tensor"前缀，如果后面有下划线也去掉
                distribution = pt_path[6:]  # 去掉"tensor"前缀
                if distribution.startswith("_"):
                    distribution = distribution[1:]  # 去掉开头的下划线
                base_save_path = get_save_path()
                if not base_save_path or not os.path.exists(base_save_path):
                    return None
                distribution = os.path.join(base_save_path, distribution)
                distribution = glob.glob(distribution)
                if distribution:
                    distribution = distribution[0]
                    distribution = os.path.relpath(distribution, base_save_path)
                    tensor_info["distribution"] = distribution

        return tensor_info

    def _parse_size_dtype(self, size_dtype_str):
        """分离size和dtype信息"""
        import re

        # 先定义已知的dtype模式，按长度排序（长的先匹配）
        # 使用末尾匹配，确保准确识别dtype
        dtype_patterns = [
            r'bfloat16$', r'bf16$',          # bf16相关，必须放在f16之前
            r'float32$', r'f32$', r'float$',
            r'float16$', r'f16$', r'half$',
            r'int64$', r'i64$', r'long$',
            r'int32$', r'i32$',
            r'int8$', r'i8$',
            r'uint8$', r'u8$',
            r'bool$',
            r'f8e4m3$',
            r'f8e5m2$',
            r'float8_e4m3fn$',
            r'float8_e5m2$'
        ]

        # 尝试匹配每个dtype模式
        dtype_str = 'f32'  # 默认值
        size_str = size_dtype_str

        for pattern in dtype_patterns:
            match = re.search(pattern, size_dtype_str, re.IGNORECASE)
            if match:
                dtype_str = match.group(0).lower()
                size_str = size_dtype_str[:-len(match.group(0))]  # 从末尾移除dtype
                break

        # 如果没有找到已知的dtype，使用通用模式（从末尾开始匹配）
        if dtype_str == 'f32' and size_str == size_dtype_str:
            # 尝试通用模式：字母+数字，从字符串末尾开始
            dtype_match = re.search(r'([a-zA-Z]+\d*)$', size_dtype_str)
            if dtype_match:
                dtype_str = dtype_match.group(1).lower()
                size_str = size_dtype_str[:-len(dtype_match.group(0))]  # 从末尾移除dtype

        # 解析size
        if 'x' in size_str:
            # 多维tensor
            size_parts = size_str.split('x')
            size = []
            for part in size_parts:
                part = part.strip()
                if part and part.isdigit():
                    size.append(int(part))
        else:
            # 单维tensor或标量
            size_str = size_str.strip()
            if size_str and size_str.isdigit():
                size = [int(size_str)]
            else:
                size = [1]

        return size, dtype_str

    def _check_contiguous(self, size, strides):
        """检查tensor是否连续"""
        if not size or not strides or len(size) != len(strides):
            return True  # 默认假设连续

        # 计算预期的连续strides
        expected_strides = [1]
        for i in range(len(size) - 1, 0, -1):
            expected_strides.insert(0, expected_strides[0] * size[i])

        return strides == expected_strides

    def _parse_input_args(self, inputs_str, aten_op_name=None, inplace_params=None):
        """解析输入参数字符串，生成详细的参数信息"""
        import re

        if inplace_params is None:
            inplace_params = {}

        # 移除外层括号
        inputs_str = inputs_str.strip('()')

        # 如果是空字符串，返回空列表
        if not inputs_str.strip():
            return []

        # 分割参数 - 改进的解析逻辑
        args = []
        current_arg = ""
        paren_count = 0
        brace_count = 0
        angle_count = 0
        bracket_count = 0  # 添加方括号计数
        in_quotes = False
        quote_char = None

        for char in inputs_str:
            if char in ['"', "'"] and not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
            elif not in_quotes:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                elif char == '<':
                    angle_count += 1
                elif char == '>':
                    angle_count -= 1
                elif char == '[':  # 添加方括号处理
                    bracket_count += 1
                elif char == ']':  # 添加方括号处理
                    bracket_count -= 1
                elif char == ',' and paren_count == 0 and brace_count == 0 and angle_count == 0 and bracket_count == 0:
                    args.append(current_arg.strip())
                    current_arg = ""
                    continue

            current_arg += char

        if current_arg.strip():
            args.append(current_arg.strip())

        parsed_args = []
        for idx, arg in enumerate(args):
            # 解析参数名和值
            if ':' in arg:
                name, value = arg.split(':', 1)
                name = name.strip()
                value = value.strip()
            else:
                name = f"arg_{idx}"
                value = arg.strip()

            # 检查是否为tensor（包括简化格式）
            if ('<' in value and '>' in value and '{' in value and '}' in value) or ('%' in value and '<' in value and '>' in value):
                tensor_info = self._parse_tensor_info(value, aten_op_name=aten_op_name, name=name, is_output=False)
                if tensor_info:
                    # 检查这个参数是否是inplace的
                    is_inplace = (
                            name in inplace_params or
                            f'idx_{idx}' in inplace_params or
                            self._is_inplace_arg_by_name(name, aten_op_name)
                    )

                    parsed_args.append({
                        "args_type": "Tensor",
                        "idx": idx,
                        "isWrite": is_inplace,
                        "name": name,
                        "value": tensor_info
                    })
                    continue

            # 根据值的格式推断类型
            args_type, parsed_value = self._infer_arg_type(value, param_name=name)

            parsed_args.append({
                "args_type": args_type,
                "idx": idx,
                "isWrite": False,
                "name": name,
                "value": parsed_value
            })

        return parsed_args

    def _infer_arg_type(self, value, param_name=None):
        """根据值的格式推断参数类型"""
        import re

        # 特殊处理：如果参数名是dtype，优先识别为dtype类型
        if param_name == "dtype":
            # 处理两种格式：
            # 1. dtype:torch.float32 (没有类型标注)
            # 2. dtype:torch_float32:dtype (有类型标注)
            if ':' in value:
                parts = value.rsplit(':', 1)
                if len(parts) == 2:
                    actual_value, type_hint = parts
                    actual_value = actual_value.strip()
                    type_hint = type_hint.strip()

                    if type_hint == 'dtype':
                        # 格式：torch_float32:dtype
                        normalized_value = self._normalize_dtype_value(actual_value)
                        return "dtype", normalized_value
                    else:
                        # 可能是其他格式，尝试按dtype处理
                        normalized_value = self._normalize_dtype_value(value)
                        return "dtype", normalized_value
                else:
                    # 格式：torch.float32 (直接值)
                    normalized_value = self._normalize_dtype_value(value)
                    return "dtype", normalized_value
            else:
                # 没有冒号，直接是dtype值
                normalized_value = self._normalize_dtype_value(value)
                return "dtype", normalized_value

        # 检查是否有类型标注（如 "-1:int", "torch_float32:dtype"）
        if ':' in value:
            parts = value.rsplit(':', 1)
            if len(parts) == 2:
                actual_value, type_hint = parts
                actual_value = actual_value.strip()
                type_hint = type_hint.strip()

                # 根据类型标注返回相应类型
                if type_hint == 'int':
                    try:
                        return "int", int(actual_value)
                    except ValueError:
                        return "str", value
                elif type_hint == 'float':
                    try:
                        return "float", float(actual_value)
                    except ValueError:
                        return "str", value
                elif type_hint == 'bool':
                    return "bool", actual_value.lower() == 'true'
                elif type_hint == 'dtype':
                    normalized_value = self._normalize_dtype_value(actual_value)
                    return "dtype", normalized_value
                elif type_hint.startswith('List['):
                    return type_hint, value
                elif type_hint.startswith('Optional['):
                    return type_hint, value
                else:
                    return type_hint, value

        # 没有类型标注，根据值本身推断
        if value.lower() in ['true', 'false']:
            return "bool", value.lower() == 'true'
        elif value.lower() == 'none':
            return "Optional[int]", None
        elif self._is_number(value):
            # 改进的数字类型推断逻辑
            try:
                # 先尝试转换为float（支持科学计数法）
                float_val = float(value)
                # 检查是否为整数
                if float_val.is_integer() and 'e' not in value.lower() and 'E' not in value.lower():
                    # 如果是整数且不是科学计数法，返回int
                    return "int", int(float_val)
                else:
                    # 否则返回float
                    return "float", float_val
            except ValueError:
                return "str", value
        elif value.startswith('[') and value.endswith(']'):
            # 尝试解析列表
            parsed_list = self._parse_list_value(value)
            if parsed_list is not None:
                # 检查列表元素类型
                if all(isinstance(x, int) for x in parsed_list):
                    return "List[int]", parsed_list
                elif all(isinstance(x, float) for x in parsed_list):
                    return "List[float]", parsed_list
                else:
                    return "List", parsed_list
            else:
                return "str", value
        else:
            return "str", value

    def _normalize_dtype_value(self, dtype_value):
        """统一dtype值的格式"""
        # 将torch_float32格式转换为torch.float32格式
        dtype_map = {
            'torch_float32': 'torch.float32',
            'torch_float16': 'torch.float16',
            'torch_bfloat16': 'torch.bfloat16',
            'torch_float64': 'torch.float64',
            'torch_double': 'torch.float64',
            'torch_int32': 'torch.int32',
            'torch_int64': 'torch.int64',
            'torch_long': 'torch.int64',
            'torch_int16': 'torch.int16',
            'torch_short': 'torch.int16',
            'torch_int8': 'torch.int8',
            'torch_uint8': 'torch.uint8',
            'torch_bool': 'torch.bool',
            'torch_complex64': 'torch.complex64',
            'torch_complex128': 'torch.complex128',
        }

        # 如果已经是标准格式，直接返回
        if dtype_value.startswith('torch.'):
            return dtype_value

        # 尝试映射转换
        return dtype_map.get(dtype_value, dtype_value)

    def _parse_list_value(self, list_str):
        """解析列表字符串，如 '[512, 64]' 或 '[1.5, 2.0]'"""
        import re

        try:
            # 移除外层方括号
            content = list_str.strip('[]')
            if not content.strip():
                return []

            # 分割元素，处理可能的空格
            elements = [elem.strip() for elem in content.split(',')]
            parsed_elements = []

            for elem in elements:
                elem = elem.strip()
                if not elem:
                    continue

                # 尝试解析为数字
                if self._is_number(elem):
                    try:
                        # 先尝试转换为float
                        float_val = float(elem)
                        # 如果是整数，转换为int
                        if float_val.is_integer() and 'e' not in elem.lower() and 'E' not in elem.lower():
                            parsed_elements.append(int(float_val))
                        else:
                            parsed_elements.append(float_val)
                    except ValueError:
                        # 如果转换失败，保持为字符串
                        parsed_elements.append(elem)
                else:
                    # 非数字，保持为字符串
                    parsed_elements.append(elem)

            return parsed_elements

        except Exception as e:
            # 解析失败，返回None
            return None

    def _is_number(self, s):
        """检查字符串是否为数字"""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def _parse_outputs(self, outputs, aten_op_name=None):
        """解析输出信息"""
        parsed_outputs = []

        if isinstance(outputs, list):
            for idx, output in enumerate(outputs):
                if isinstance(output, str):
                    # 检查是否为tensor格式
                    if ('<' in output and '>' in output and '{' in output and '}' in output) or ('%' in output and '<' in output and '>' in output):
                        tensor_info = self._parse_tensor_info(output, aten_op_name=aten_op_name, name=f"out{idx}", is_output=True)
                        if tensor_info:
                            parsed_outputs.append({
                                "args_type": "Tensor",
                                "idx": idx,
                                "name": f"out{idx}",
                                "value": tensor_info
                            })
                            continue

                    # 非tensor格式，推断类型
                    args_type, parsed_value = self._infer_arg_type(output)
                    parsed_outputs.append({
                        "args_type": args_type,
                        "idx": idx,
                        "name": f"out{idx}",
                        "value": parsed_value
                    })
                else:
                    parsed_outputs.append({
                        "args_type": "unknown",
                        "idx": idx,
                        "name": f"out{idx}",
                        "value": str(output)
                    })
        else:
            if isinstance(outputs, str):
                # 检查是否为tensor格式
                if ('<' in outputs and '>' in outputs and '{' in outputs and '}' in outputs) or ('%' in outputs and '<' in outputs and '>' in outputs):
                    tensor_info = self._parse_tensor_info(outputs, aten_op_name=aten_op_name, name="out0", is_output=True)
                    if tensor_info:
                        parsed_outputs.append({
                            "args_type": "Tensor",
                            "idx": 0,
                            "name": "out0",
                            "value": tensor_info
                        })
                    else:
                        parsed_outputs.append({
                            "args_type": "str",
                            "idx": 0,
                            "name": "out0",
                            "value": outputs
                        })
                else:
                    # 非tensor格式，推断类型
                    args_type, parsed_value = self._infer_arg_type(outputs)
                    parsed_outputs.append({
                        "args_type": args_type,
                        "idx": 0,
                        "name": "out0",
                        "value": parsed_value
                    })
            else:
                parsed_outputs.append({
                    "args_type": "unknown",
                    "idx": 0,
                    "name": "out0",
                    "value": str(outputs)
                })

        return parsed_outputs

    def _get_op_schema(self, op_name):
        """获取操作的schema信息"""
        # 尝试从PyTorch中获取真实的schema
        real_schema = self._get_real_op_schema(op_name)
        if real_schema:
            return real_schema

        # 如果无法获取真实schema，使用硬编码的映射作为后备
        schema_map = {
            'aten::add': 'aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor',
            'aten::mul': 'aten::mul.Tensor(Tensor self, Tensor other) -> Tensor',
            'aten::matmul': 'aten::matmul(Tensor self, Tensor other) -> Tensor',
            'aten::_log_softmax': 'aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor',
            'aten::_log_softmax.out': 'aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)',
            'aten::relu': 'aten::relu(Tensor self) -> Tensor',
            'aten::conv2d': 'aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor',
            'aten::linear': 'aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor',
            'xformers_flash::flash_fwd': 'xformers_flash::flash_fwd(Tensor query, Tensor key, Tensor value, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, Tensor? seqused_k, int max_seqlen_q, int max_seqlen_k, float p, float softmax_scale, bool is_causal, int window_left, int window_right, bool return_softmax, Tensor? block_tables) -> (Tensor, Tensor, Tensor)',
        }

        return schema_map.get(op_name, f"{op_name}(...) -> (...)")

    def _get_real_op_schema(self, op_name):
        """从PyTorch中获取真实的算子schema"""
        try:
            import torch

            # 尝试通过torch._C._get_schema获取schema
            if hasattr(torch._C, '_get_schema'):
                schema = torch._C._get_schema(op_name)
                print("1111111111111111111schema", schema)
                if schema:
                    return str(schema)

            # 尝试通过torch.ops获取schema
            if '::' in op_name:
                namespace, op = op_name.split('::', 1)
                if hasattr(torch.ops, namespace):
                    namespace_ops = getattr(torch.ops, namespace)
                    if hasattr(namespace_ops, op):
                        op_func = getattr(namespace_ops, op)
                        if hasattr(op_func, '_schema'):
                            print("4444444444444444444444444schema", op_func._schema)
                            return str(op_func._schema)

            # 尝试通过torch._C._jit_get_schema获取schema
            if hasattr(torch._C, '_jit_get_schema'):
                try:
                    schema = torch._C._jit_get_schema(op_name)
                    print("2222222222222222222schema", schema)
                    if schema:
                        return str(schema)
                except:
                    pass

            # 尝试通过torch._C._get_operator_schema获取schema
            if hasattr(torch._C, '_get_operator_schema'):
                try:
                    schema = torch._C._get_operator_schema(op_name)
                    print("3333333333333333333schema", schema)
                    if schema:
                        return str(schema)
                except:
                    pass

        except Exception as e:
            pass

        return None

    def debug_print_ops_status(self):
        """调试方法：打印当前存储的所有操作状态"""
        pass

    def _get_aten_name(self, op_name):
        """获取操作对应的aten名称"""
        # 这里可以添加映射逻辑
        # 暂时返回一个简单的转换
        name_map = {
            'xformers_flash::flash_fwd': '_atenFlashAttention',
            'aten::add': '_atenAdd',
            'aten::mul': '_atenMul',
            'aten::matmul': '_atenMatmul',
            'aten::_log_softmax': '_atenLogSoftmax'
        }
        return name_map.get(op_name, f"_aten{op_name.split('::')[-1].title()}")

    def next_iter(self):
        self.iter += 1

    def next_step(self):
        self.step += 1
        self.iter = 0

    def add(self, attr, key, data):
        if key not in getattr(self, attr):
            getattr(self, attr)[key] = {data}

    def clear(self):
        self.iter = 0
        self.model = {}
        self.graph = {}
        self.module = {}
        self.ops = {}
        self.op_info = {}
        self.comms = {}
        self.apis = {}
        self.npz_num = 0
        self.state_stack = []
        self.api_stack = []
        # 清理新增的数据结构
        self.ops_detailed = {}
        self.op_schemas = {}
        self.op_aten_map = {}

    def get_fw_version(self, op):
        if op.startswith('te'):
            return self.fw_version['te']
        else:
            return f'torch.{pt_version}'

    def trace(self, log, bak=False):
        tab = len(self.state_stack) + len(self.api_stack)
        if bak and tab > 0:
            tab -= 1
        state = f'{self.step}_{self.iter}'
        if state not in self.model:
            self.model[state] = []
        self.model[state].append('  ' * tab + log)

    def add_module(self, stage, state, name, module_name, inputs, outs):
        if inputs is None:
            args, kwargs = [], {}
        elif len(inputs) == 2:
            args, kwargs = inputs
        else:
            args, kwargs = inputs, {}
        if state == 'call':
            inputs = f"{args}, {kwargs}".strip(', ')
            self.trace(f'- {self.step}_{self.iter}_{stage}_module::{module_name}:', True)
            self.trace(f'- param_name: {name}')
            self.trace(f'- inputs: {inputs}')
            # self.trace(f'  - outputs: {outs}')
        else:
            inputs = self.add_to_dict('module', f'module::{module_name}', args, kwargs, outs, {'param_name': name})

    def get_item_name(self, args, kwargs, outs):
        return f"({outs})=f({args},{kwargs})".replace("'", "").replace(" ", "")

    def add_comm(self, name, args, kwargs, outs):
        if name not in self.comms:
            self.comms[name] = []
        self.comms[name].append((args, kwargs, outs, self.state))

    def add_to_dict(self, attr, name, args, kwargs, outs, conf=None, real_schema=None):
        _dict = getattr(self, attr)
        if name not in _dict:
            _dict[name] = dict()
        item = self.get_item_name(args, kwargs, outs)
        inputs = f"{args}, {kwargs}".strip(', ')
        if item not in _dict[name]:
            if "NVTE_Fused_Attn_Backend" in str(type(outs)) \
                    or "torch.classes.c10d.Work" in str(type(outs)) \
                    or "torch.classes.c10d.ProcessGroup" in str(type(outs)):
                outs = str(outs)

            if isinstance(outs, list) and outs:
                for i, out in enumerate(outs):
                    if "torch.ScriptObject" in str(type(out)):
                        outs[i] = str(out)

            _dict[name][item] = {
                'idx': len(_dict[name]),
                'use_count': 1,
                'inputs': f"({inputs})",
                'outputs': outs,
                'state': set(),
            }
            if conf:
                for k, v in conf.items():
                    _dict[name][item][k] = v
        else:
            _dict[name][item]['use_count'] += 1
        _dict[name][item]['state'].add(str(self.state))

        # 为目标格式准备详细信息
        if attr == 'ops':
            self._store_detailed_op_info(name, item, inputs, outs, _dict[name][item], real_schema=real_schema)

        return f"({inputs})"

    def _store_detailed_op_info(self, op_name, item_key, inputs, outputs, item_info, real_schema=None):
        """存储详细的操作信息用于目标格式"""

        if op_name not in self.ops_detailed:
            # 如果有真实的schema信息，使用它；否则使用默认的获取方法
            schema = real_schema if real_schema else self._get_op_schema(op_name)

            self.ops_detailed[op_name] = {
                'aten_op_name': op_name,
                'dynamic_op': False,  # 简化处理
                'params': [],
                'params_count': 0,
                'schema': schema,
                'aten_name': self._get_aten_name(op_name),
                'use_count': 0,
                'items': {}
            }
        else:
            # 操作已存在，检查是否需要更新schema
            existing_schema = self.ops_detailed[op_name]['schema']

            # 如果当前有真实的schema，且存储的是默认schema，则更新
            if real_schema and real_schema != existing_schema:
                # 检查存储的schema是否是默认格式（包含"(...) -> (...)"模式）
                if "(...) -> (...)" in existing_schema:
                    self.ops_detailed[op_name]['schema'] = real_schema

        # 解析inplace信息
        current_schema = self.ops_detailed[op_name]['schema']
        inplace_params = self._parse_inplace_from_schema(current_schema)

        if item_key not in self.ops_detailed[op_name]['items']:
            # 解析输入参数（传递inplace信息）
            parsed_inputs = self._parse_input_args(inputs, aten_op_name=op_name, inplace_params=inplace_params)
            # 解析输出
            parsed_outputs = self._parse_outputs(outputs, aten_op_name=op_name)

            self.ops_detailed[op_name]['items'][item_key] = {
                'idx': len(self.ops_detailed[op_name]['items']) + 1,
                'inputs': parsed_inputs,
                'outputs': parsed_outputs,
                'use_count': 1
            }
            self.ops_detailed[op_name]['params'].append(self.ops_detailed[op_name]['items'][item_key])
        else:
            self.ops_detailed[op_name]['items'][item_key]['use_count'] += 1

        # 更新总的使用次数
        self.ops_detailed[op_name]['use_count'] += 1
        self.ops_detailed[op_name]['params_count'] = len(self.ops_detailed[op_name]['items'])

    def _parse_inplace_from_schema(self, schema_str):
        """解析schema字符串，提取inplace参数信息"""
        import re

        if not schema_str:
            return {}

        inplace_params = {}

        try:
            # 解析schema格式: op_name(arg1_type arg1_name, arg2_type arg2_name, ...) -> return_type
            # 提取参数部分
            match = re.search(r'\((.*?)\)\s*->', schema_str)
            if not match:
                return {}

            params_str = match.group(1)
            if not params_str.strip():
                return {}

            # 分割参数，处理嵌套的括号和泛型
            params = []
            current_param = ""
            paren_count = 0
            bracket_count = 0

            for char in params_str:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                elif char == ',' and paren_count == 0 and bracket_count == 0:
                    params.append(current_param.strip())
                    current_param = ""
                    continue

                current_param += char

            if current_param.strip():
                params.append(current_param.strip())

            # 分析每个参数
            for idx, param in enumerate(params):
                param = param.strip()
                if not param:
                    continue

                # 检查是否有inplace标记 (!)
                # 例如: Tensor(a!) self, Tensor(b!) other
                is_inplace = '!' in param

                # 提取参数名
                # 通常格式是: type_name param_name 或者 type_name(annotation) param_name
                param_name = None

                # 先处理带括号的类型注释
                if '(' in param and ')' in param:
                    # 去掉类型注释，提取参数名
                    param_without_annotation = re.sub(r'\([^)]*\)', '', param)
                    parts = param_without_annotation.strip().split()
                    if len(parts) >= 2:
                        param_name = parts[-1]  # 最后一个通常是参数名
                else:
                    # 简单的格式：type param_name
                    parts = param.split()
                    if len(parts) >= 2:
                        param_name = parts[-1]

                if param_name and is_inplace:
                    inplace_params[param_name] = True
                    inplace_params[f'idx_{idx}'] = True  # 也通过索引记录

        except Exception as e:
            pass

        return inplace_params

    def _is_inplace_arg_by_name(self, arg_name, op_name):
        """基于参数名和操作名判断是否为inplace参数"""
        if not arg_name or not op_name:
            return False

        # 一些常见的inplace操作模式
        inplace_patterns = [
            # 以下划线结尾的操作通常是inplace的，如add_, mul_等
            '_',
            '.out',  # 带.out的操作通常第一个参数是输出tensor
        ]

        # 检查操作名是否表明这是inplace操作
        for pattern in inplace_patterns:
            if op_name.endswith(pattern):
                # 对于以_结尾的操作，通常第一个参数（self）是被修改的
                if pattern == '_' and arg_name in ['self', 'input']:
                    return True
                # 对于.out操作，通常有个out参数
                elif pattern == '.out' and arg_name == 'out':
                    return True

        # 一些特殊的inplace操作名称检查
        special_inplace_ops = {
            'aten::add_': ['self'],
            'aten::mul_': ['self'],
            'aten::div_': ['self'],
            'aten::sub_': ['self'],
            'aten::copy_': ['self'],
            'aten::fill_': ['self'],
            'aten::zero_': ['self'],
            'aten::uniform_': ['self'],
            'aten::normal_': ['self'],
        }

        if op_name in special_inplace_ops:
            return arg_name in special_inplace_ops[op_name]

        return False

    def _get_additional_op_info(self, func):
        """获取算子的额外信息"""
        info = {}

        # 检查常见属性
        attrs_to_check = [
            '_schema',
            'schema',
            '_overloadpacket',
            'overload_name',
            'name',
            'packet'
        ]

        for attr in attrs_to_check:
            if hasattr(func, attr):
                try:
                    value = getattr(func, attr)
                    info[attr] = str(value)
                except Exception as e:
                    pass

        # 如果有schema对象，尝试获取更详细的信息
        if hasattr(func, '_schema') and func._schema:
            try:
                schema = func._schema
                if hasattr(schema, 'arguments'):
                    for i, arg in enumerate(schema.arguments):
                        if hasattr(arg, 'annotation') and arg.annotation:
                            # 检查参数注释中是否有可变性信息
                            annotation = str(arg.annotation)
                            if '!' in annotation:
                                info[f'arg_{i}_inplace'] = True
            except Exception as e:
                pass

        return info

    def _get_schema_from_torch_dispatch(self, func):
        """从torch_dispatch中的func对象获取schema信息"""
        try:
            if hasattr(func, '_schema'):
                schema = func._schema
                # 直接返回schema的字符串表示
                schema_str = str(schema)

                # 解析inplace信息
                inplace_params = self._parse_inplace_from_schema(schema_str)

                # 获取额外的算子信息
                additional_info = self._get_additional_op_info(func)

                return schema_str
            else:
                return None
        except Exception as e:
            pass

        return None

    def set_step(self, step=None, state=None, profile=True):
        if GLOBAL_CFG.get('perf', False):
            return None
        GLOBAL_CFG['step_by_customer'] = True
        if state == 'start':
            if GLOBAL_CFG['nvtx'] and profile and self.rank in GLOBAL_CFG['nvtx_args']['ranks'] and step == GLOBAL_CFG['nvtx_args']['start']:
                torch.cuda.cudart().cudaProfilerStart()
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
            if step is None:
                self.next_step()
            else:
                self.step = step
                self.iter = 0
            torch.cuda.nvtx.range_push(f"trace::step:{self.step}")
        elif state == 'end':
            if GLOBAL_CFG['nvtx'] and profile and self.rank in GLOBAL_CFG['nvtx_args']['ranks'] and step == GLOBAL_CFG['nvtx_args']['end']:
                torch.cuda.cudart().cudaProfilerStop()
            torch.cuda.nvtx.range_pop()

    def set_iter(self, iter=None, state=None):
        if GLOBAL_CFG.get('perf', False):
            return None
        GLOBAL_CFG['iter_by_customer'] = True
        if state == 'start':
            if iter is None:
                self.next_iter()
            else:
                self.iter = iter

    def add_api(self, name, state, args, kwargs, outs, func_file):
        if state == 'call':
            inputs = f"{args}, {kwargs}".strip(', ')
            self.trace(f'- name: api::{name}')
            self.trace(f'  inputs: {inputs}')
            # self.trace(f'  outputs: {outs}')
            self.trace(f'  file: {func_file}')
        else:
            inputs = self.add_to_dict('apis', f'api::{name}', args, kwargs, outs, {'file': func_file})

    def add_op(self, name, device_id, need_save, args, kwargs, outs, op_desc, real_schema=None):
        if name not in self.ops:
            self.op_info[name] = {}
        conf = {'support_by_backend': True} if name in self.support_by_torch else None
        inputs = self.add_to_dict('ops', name, args, kwargs, outs, conf, real_schema=real_schema)
        if need_save:
            # print('op_info:', name, op_desc, len(self.ops[name]))
            if op_desc in self.op_info[name]:
                self.op_info[name][op_desc].append(len(self.ops[name]))
            else:
                self.op_info[name][op_desc] = [len(self.ops[name])]
            self.save_trace_info()
        formatted_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
        stream = current_stream() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
        self.trace(f'{formatted_time} %{pid}:{device_id}:{stream}% {name}{inputs} -> {outs}')

    def trace_optimizer(self, name, module_name, state):
        self.state = 'optimizer'
        # if state == 'OptimStep':
        #     self.next_step()
        f_name = '{}_n{}_rank{}'.format(module_name, self.step, self.rank)
        self.print_msg(name, f_name, state)

    def new_model(self, names, model):
        pass

    def trace_value(self, val, need_save=False):
        parser = parser_map.get(class_name(val), default_parser)
        trace_outs, trace_str = parser(val, save=need_save)
        outs_str = trace_outs.getString()
        if isinstance(trace_str, str) and any(k in trace_str for k in ['torch_autograd', 'Function_apply', 'Function_apply', 'torch__C']):
            trace_str = val
        return outs_str, trace_str

    def trace_list(self, args, need_save=False):
        saved_args = []
        args_str = ""
        for i, tensor in enumerate(args):
            need_arg = need_save + '_' + str(i) if need_save and isinstance(need_save, str) else need_save
            trace_arg, trace_str = self.trace_value(tensor, need_arg)
            saved_args.append(str(trace_str))
            args_str += f"{trace_arg}, "
        args_str = args_str[:-2]
        return args_str, ", ".join(saved_args)

    def trace_list_default(self, args, default_args, need_save=False):
        saved_kwargs = []
        args_str = ""
        m = len(args)
        for i, item in enumerate(default_args):
            key, default_arg = item
            arg = args[i] if i < m else default_arg
            need_arg = need_save + '_' + str(key) if need_save and isinstance(need_save, str) else need_save
            trace_arg, trace_str = self.trace_value(arg, need_arg)
            if i < m:
                saved_kwargs.append(f"{key}:{trace_str}")
            args_str += f"{trace_arg}, "
        args_str = args_str[:-2]
        return args_str, ", ".join(saved_kwargs)

    def trace_dict(self, kwargs, startswith='', need_save=False):
        saved_kwargs = []
        kwargs_str = startswith
        for key, tensor in kwargs.items():
            need_arg = need_save + '_' + str(key) if need_save and isinstance(need_save, str) else need_save
            trace_arg, trace_str = self.trace_value(tensor, need_arg)
            saved_kwargs.append(f"{key}:{trace_str}")
            kwargs_str += f"{key}={trace_arg}, "
        kwargs_str = kwargs_str[:-2]
        return kwargs_str, ", ".join(saved_kwargs)

    def trace_dict_default(self, kwargs, default_kwargs, startswith='', need_save=False):
        saved_kwargs = []
        kwargs_str = startswith
        for i, key in enumerate(default_kwargs):
            value = kwargs[key] if key in kwargs else default_kwargs[key]
            need_arg = need_save + '_' + str(key) if need_save and isinstance(need_save, str) else need_save
            trace_value, trace_str = self.trace_value(value, need_arg)
            if key in kwargs:
                saved_kwargs.append(f"{key}:{trace_str}")
            kwargs_str += f"{key}={trace_value}, "
        kwargs_str = kwargs_str[:-2]
        return kwargs_str, ", ".join(saved_kwargs)

    def trace_module(self, stage, state, iter, module_name, name, module, args, kwargs, out, get_data, model_iter=False):
        f_name = '{}_{}_{}_n{}_rank{}'.format(name, module_name, state, iter, self.rank)
        if state == 'call':
            if stage == 'fwd':
                if self.state.startswith('start'):
                    if not GLOBAL_CFG.get('step_by_customer'):
                        self.next_step()
                elif not GLOBAL_CFG.get('iter_by_customer'):
                    if model_iter:
                        self.next_iter()
                    # elif (self.state.startswith('bwd') or self.state.startswith('optimizer')):
                    #     print('next_iter check: need to check', stage, state, iter, module_name, name)
                    #     print(self.state)
                    #     exit()
            res = get_data(name, f_name, module, None, args, kwargs)
            self.state_stack.append(name + '_' + module_name)
            self.state = stage + ':' + self.state_stack[-1] if self.state_stack else 'root'
            self.add_module(stage, state, name, module_name, res['inputs_str'], '')
            self.print_msg(name, f_name, 'pre_' + stage)
            ### modified
            if GLOBAL_CFG.get('module_trace') and (name in GLOBAL_CFG['module_trace']['module']
                                                   or module_name in GLOBAL_CFG['module_trace']['cls']):
                if GLOBAL_CFG['module_trace'].get('max') and GLOBAL_CFG['module_trace']['max'] <= self.step:
                    return
                if module_name == "Attention" and GLOBAL_CFG.get('dump_kv_cache_enable', False):
                    from vllm.forward_context import get_forward_context
                    from vllm.config import get_current_vllm_config
                    forward_context = get_forward_context()
                    vllm_config = get_current_vllm_config()
                    attn_meta = {}
                    for k in ['num_prefills', 'num_prefill_tokens', 'num_decode_tokens', 'max_prefill_seq_len',
                                'max_decode_seq_len', 'max_query_len', 'max_decode_query_len', 'slot_mapping',
                                'input_positions', 'multi_modal_placeholder_index_maps', 'enable_kv_scales_calculation'
                                ]:
                        attn_meta[k] = getattr(forward_context.attn_metadata, k)
                    kv_cache = module.kv_cache[forward_context.virtual_engine]
                    self_kv_cache = kv_cache.view(-1, kv_cache.shape[-1])[:attn_meta['slot_mapping'][-1], :]
                    res['init']['kv_cache'] = self_kv_cache
                    res['init']['attn_meta'] = attn_meta
                    res['init']['block_size'] = vllm_config.cache_config.block_size
                self.save_pt(stage, res)
        else:
            self.print_msg(name, f_name, stage)
            self.state = stage + ':' + self.state_stack[-1] if self.state_stack else 'root'
            if len(self.state_stack) > 0:
                last_module = self.state_stack.pop(-1)
                if name + '_' + module_name != last_module:
                    if len(self.state_stack) > 0 and self.state_stack[-1] == name + '_' + module_name:
                        self.state_stack.pop(-1)
                        # print('Warning module return Loss:', last_module, name + '_' + module_name)
                    else:
                        pass
                        # print('Err module return:', last_module, name + '_' + module_name)
                # exit() need to check multi stream status if print Err.
            res = get_data(name, f_name, module, out, args, kwargs)
            self.add_module(stage, state, name, module_name, res['inputs_str'], res['outputs_str'])
            if GLOBAL_CFG.get('module_trace') and (name in GLOBAL_CFG['module_trace']['module']
                                                   or module_name in GLOBAL_CFG['module_trace']['cls']):
                if GLOBAL_CFG['module_trace'].get('max') and GLOBAL_CFG['module_trace']['max'] <= self.step:
                    return
                self.save_pt(stage, res)

    def trace_api(self, func_name, state, device_id, func_file='', args=[], kwargs={}, outs=None, cache=None):
        if GLOBAL_CFG['sync_mode']:
            try:
                torch.cuda.synchronize()
            except:
                pass
        if cache is None:
            args_str, saved_args = self.trace_list(args)
            kwargs_str, saved_kwargs = self.trace_dict(kwargs)
            op_func = f"{func_name}({args_str}{kwargs_str})"
        else:
            op_func, saved_args, saved_kwargs = cache
        if state == 'call':
            self.add_api(func_name, state, saved_args, saved_kwargs, '', func_file)
            self.api_stack.append(func_name)
        else:
            last_func =  self.api_stack.pop(-1)
            if outs is None:
                outs_str, saved_outs = '', ''
            else:
                outs_str, saved_outs = self.trace_value(outs)
                outs_str += '='
            self.add_api(func_name, state, saved_args, saved_kwargs, saved_outs, func_file)
            if func_name != last_func:
                print('Err api return:', last_func, func_name)
                exit()
        if GLOBAL_CFG['print_log']:
            formatted_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            stream = current_stream() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
            print(f"{formatted_time} api_{state}: {OPTRACE_VERSION}{pid}:{device_id}:{stream} {outs_str}torch.{pt_version}.{op_func} \n", end='', flush=True)
        return op_func, saved_args, saved_kwargs

    def trace_op(self, op_name, state, device_id, need_save=False, args=[], kwargs={}, default_args=None, default_kwargs=None, outs=None, cache=None, real_schema=None):
        if GLOBAL_CFG['sync_mode'] and self.state != 'start':
            torch.cuda.synchronize()
        save_cfg = f'{self.state}_{self.step}_{self.iter}_{op_name}' if need_save else False
        if cache is None:
            if default_args:
                args_str, saved_args = self.trace_list_default(args, default_args, save_cfg)
            else:
                args_str, saved_args = self.trace_list(args, save_cfg)
            if default_kwargs:
                kwargs_str, saved_kwargs = self.trace_dict_default(kwargs, default_kwargs, need_save=save_cfg)
            else:
                kwargs_str, saved_kwargs = self.trace_dict(kwargs, need_save=save_cfg)
            op_func = f"{op_name}({args_str}{kwargs_str})"
        else:
            op_func, saved_args, saved_kwargs = cache
        if state == 'call':
            if 'Optimizer' in saved_args:
                self.state = 'optimizer'
                if not GLOBAL_CFG.get('step_by_customer'):
                    self.next_step()
            if op_name == 'flash_attn::varlen_fwd':
                if GLOBAL_CFG.get('customer_kv_cache_check'):
                    print('prepare:', self.state)
                    kv_for_send, k, v = prepare_kv(self.state, *args, **kwargs)
                    if kv_for_send[0]:
                        print('\tsend to gpu')
                        customer_kv_cache.add_rsv(kv_for_send)
                        print('\tget_kv on gpu')
                        k_from_cache, v_from_cache = customer_kv_cache.get_kv(0, kv_for_send[0])
                        print('\tcompare with gpu')
                        if torch.isnan(k.cpu()).any() or torch.isinf(k.cpu()).any():
                            pass
                        else:
                            if not torch.allclose(k.cpu(), k_from_cache.cpu()):
                                print('\tk:', k.cpu())
                                print('\tk_from_cache:', k_from_cache.cpu())
                            if not torch.allclose(v.cpu(), v_from_cache.cpu()):
                                print('\tv:', k.cpu())
                                print('\tv_from_cache:', k_from_cache.cpu())
                            assert torch.allclose(k.cpu(), k_from_cache.cpu())
                            assert torch.allclose(v.cpu(), v_from_cache.cpu())
                        print('verified')
                elif GLOBAL_CFG.get('fallback_gpu') and op_name in GLOBAL_CFG['fallback_gpu']:
                    print('prepare:', self.state)
                    state_id, args_for_send, kwargs_for_send = prepare_for_send(self.state, *args, **kwargs)
                    if args_for_send or kwargs_for_send:
                        args, kwargs, fallback_result, fallback_flag = self.fallback_service.exec(op_name, state_id, self.rank, args_for_send, kwargs_for_send)
                    else:
                        args, kwargs, fallback_result, fallback_flag = self.fallback_service.exec(op_name, 0, self.rank, args, kwargs)
                    print('\texec by gpu')
                    return op_func, saved_args, saved_kwargs, fallback_result, fallback_flag
            # elif op_name == 'profiler::_record_function_enter_new':
            #     print('op_name', op_name)
            #     print('saved_args', saved_args)
            #     print('saved_kwargs', saved_kwargs)
            #     print('op_func', op_func)
        else:
            save_cfg = save_cfg + '_out' if need_save else False
            if isinstance(outs, (list, tuple)):
                outs_str, saved_outs = self.trace_list(outs, save_cfg)
            elif isinstance(outs, dict):
                outs_str, saved_outs = self.trace_dict(outs, need_save=save_cfg)
            else:
                outs_str, saved_outs = self.trace_value(outs, save_cfg)
            self.add_op(op_name, device_id, need_save, saved_args, saved_kwargs, saved_outs, op_func, real_schema=real_schema)
        if GLOBAL_CFG['print_log']:
            if outs is None:
                outs_str = ''
            else:
                outs_str += '='
            formatted_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            stream = current_stream() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
            print(f"{formatted_time} dispatch_{state}: {OPTRACE_VERSION}{pid}:{device_id}:{stream} {outs_str}torch.{pt_version}.{op_func}\n", end='', flush=True)
        return op_func, saved_args, saved_kwargs

    def save_trace_info(self):
        save_path = get_save_path()
        with open(os.path.join(save_path, 'model_ops.json'), "w") as f:
            json.dump(self.op_info, f, indent=4)
        try:
            torch.save(self.ops, os.path.join(save_path, 'model_trace.pt'))
        except Exception as e:
            print(f"Ops: {self.ops}")
            raise e
        if GLOBAL_CFG.get('op_trace'):
            op_test = {'data_path': os.path.abspath(save_path),
                       'model_trace': 'model_trace.pt',
                       'op_count': len(self.op_info),
                       'ops': self.op_info
                       }
            with open(os.path.join(save_path, 'op_test.json'), "w") as f:
                json.dump(op_test, f, indent=4)

    def trace_comm(self, nccl_func, op_name, data):
        if GLOBAL_CFG['nccl_func_save_log']:
            nccl_func.save_log(op_name, data, f'{self.step}_{self.iter}', self.npz_num)
            self.npz_num += 1

    def print_msg(self, name, f_name, event):
        if GLOBAL_CFG['sync_mode']:
            torch.cuda.synchronize()
        if GLOBAL_CFG['print_log']:
            if 'pre' in event:
                stage = 'module_call'
            else:
                stage = 'module_return'
            name_info = f_name.split('_')
            module_name, layer_iter, rank = '_'.join(name_info[:-2]), name_info[-2], name_info[-1]
            formatted_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            stream = current_stream() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
            print(f"{formatted_time} {stage}: {OPTRACE_VERSION}{pid}:None:{stream} torch.{pt_version}.{module_name}({event}:{name},{layer_iter},{rank}) \n", end='', flush=True)

    def save_pt(self, stage, data):
        if data:
            save_path = get_save_path()
            path = os.path.join(save_path, data['info']['f_name'] + '_trace.pt')
            if stage == 'bwd':
                res = torch.load(path, map_location=torch.device('cpu'))
                if GLOBAL_CFG['moduel_save_state']:
                    res['state_bwd'] = data['state_dict']
                res['out_bwd'] = data['inputs']
                res['in_bwd'] = data['outputs']
                torch.save(res, path)
            else:
                torch.save(data, path)

    def summary(self, need_save):
        if len(self.ops) == 0 and len(self.model) == 0 and len(self.module) == 0 and len(self.apis) == 0:
            return
        print('save trace data:', len(self.model), len(self.module), len(self.apis), len(self.ops))

        # 生成目标格式的ops.json
        self._generate_target_ops_json()

        # 生成原始格式的文件（保持兼容性）
        for name in self.ops:
            for item in self.ops[name]:
                self.ops[name][item]['state'] = list(self.ops[name][item]['state'])
            self.ops[name] = list(self.ops[name].values())
        save_path = get_save_path()
        with open(os.path.join(save_path, 'ops_original.json'), "w") as f:
            f.write(json.dumps(self.ops, indent=4))
        for name in self.apis:
            for item in self.apis[name]:
                self.apis[name][item]['state'] = list(self.apis[name][item]['state'])
            self.apis[name] = list(self.apis[name].values())
        with open(os.path.join(save_path, 'apis.json'), "w") as f:
            f.write(json.dumps(self.apis, indent=4))
        for name in self.module:
            for item in self.module[name]:
                self.module[name][item]['state'] = list(self.module[name][item]['state'])
            self.module[name] = list(self.module[name].values())
        with open(os.path.join(save_path, 'module.json'), "w") as f:
            f.write(json.dumps(self.module, indent=4))
        for state in self.model:
            with open(os.path.join(save_path, f'{state}.trace'), "w") as f:
                f.write('\n'.join(self.model[state]))
        if need_save:
            self.save_trace_info()
        print('='*50, 'trace ops','='*50)
        for name in self.ops:
            print(name, len(self.ops[name]))
            for v in self.ops[name]:
                print(f'\t%{self.get_fw_version(name)}:{self.rank}% {name}{v["inputs"]} -> {v["outputs"]}')
        self.clear()

    def _generate_target_ops_json(self):
        """生成目标格式的ops.json文件"""
        save_path = get_save_path()

        # 构建目标格式的数据
        target_ops = []

        for op_name, op_info in self.ops_detailed.items():
            # 清理params列表，移除临时的items字段
            clean_params = []
            for param in op_info['params']:
                clean_param = {
                    'idx': param['idx'],
                    'inputs': param['inputs'],
                    'outputs': param['outputs'],
                    'use_count': param['use_count']
                }
                clean_params.append(clean_param)

            target_op = {
                'aten_op_name': op_info['aten_op_name'],
                'dynamic_op': op_info['dynamic_op'],
                'params': clean_params,
                'params_count': op_info['params_count'],
                'schema': op_info['schema'],
                'aten_name': op_info['aten_name'],
                'use_count': op_info['use_count']
            }

            target_ops.append(target_op)

        # 写入目标格式的ops.json
        with open(os.path.join(save_path, 'ops.json'), "w") as f:
            json.dump(target_ops, f, indent=4)

        print(f'Generated target format ops.json with {len(target_ops)} operations')

    def start_fallback_gpu(self):
        if GLOBAL_CFG.get('fallback_gpu'):
            self.fallback_service.start()

    def stop_fallback_gpu(self):
        if self.fallback_service:
            self.fallback_service.stop()

    def set_fallback_gpu_debug(self, func):
        if GLOBAL_CFG.get('fallback_gpu'):
            self.fallback_service.config_as_debug(func)


tracker = Tracker()
