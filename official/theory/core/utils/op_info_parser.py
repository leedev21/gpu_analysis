class Parser(object):
    def __init__(self, name, line='', out=False, debug={}):
        self.args = []
        self.kwargs = []
        self.out = out
        self.dtype_map = {
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
        self.debug = debug
        self.print_log = False
        if line:
            self.create_by_trace(line, name)

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

    def _parse_tensor_info(self, tensor_str, aten_op_name=None, name=None, is_output=False):
        """解析tensor字符串，提取详细信息"""
        if self.print_log:
            print('_parse_tensor:', tensor_str)
        import re

        # 处理两种格式：
        # 1. %107.0:<1024x16x80xbf16>{1280, 80, 1}
        # 2. <8x151936xf32>{151936, 1}

        tensor_info = {
            'tensor_id': None,
            'size_dtype_str': None,
            'stride_str': None,
            'scale': None
        }

        # 先尝试匹配完整格式
        print(tensor_str, aten_op_name, name, is_output)
        pattern_list = [(r'%(\d+\.\d+):<(.+?)>(\{.+?\})', ['tensor_id', 'size_dtype_str', 'stride_str']),
                        (r'<(.+?)>(\{.+?\})', ['size_dtype_str', 'stride_str']),
                        # (r'(.+?)\*<(.+?)>', ['scale', 'size_dtype_str']),
                        (r'<(.+?)>', ['size_dtype_str'])]
        for pattern in pattern_list:
            match = re.search(pattern[0], tensor_str)
            if match:
                for i, key in enumerate(pattern[1]):
                    tensor_info[key] = match.group(i + 1)
                print('pattern:', pattern[0])
                break
        print(tensor_info)
        if tensor_info['scale']:
            print('scale:', tensor_info['scale'], 'size_dtype_str:', tensor_info['size_dtype_str'])
            exit()
        if not tensor_info['size_dtype_str']:
            return None

        # pattern1 = r'%(\d+\.\d+):<(.+?)>(\{.+?\})'
        # match1 = re.search(pattern1, tensor_str)

        # if match1:
        #     tensor_id = match1.group(1)
        #     size_dtype_str = match1.group(2)
        #     stride_str = match1.group(3)
        # else:
        #     # 尝试匹配简化格式
        #     pattern2 = r'<(.+?)>(\{.+?\})'
        #     match2 = re.search(pattern2, tensor_str)
        #     if match2:
        #         size_dtype_str = match2.group(1)
        #         stride_str = match2.group(2)
        #     else:
        #         pattern3 = r'<(.+?)>'
        #         match3 = re.search(pattern3, tensor_str)
        #         if match3:
        #             size_dtype_str = match3.group(1)
        #             stride_str = None
        #         else:
        #             return None

        # 解析size和dtype
        size, dtype_str = self._parse_size_dtype(tensor_info['size_dtype_str'])

        # 映射dtype
        dtype = self.dtype_map.get(dtype_str, 'Float')

        # 解析stride
        strides = []
        is_contiguous = True
        if tensor_info['stride_str']:
            stride_match = re.search(r'\{(.+?)\}', tensor_info['stride_str'])
            if stride_match:
                stride_content = stride_match.group(1)
                if stride_content.strip():
                    strides = [int(x.strip()) for x in stride_content.split(',') if x.strip()]

            # 计算is_contiguous
            is_contiguous = self._check_contiguous(size, strides)

        # 构建基本的tensor信息
        tensor_info = {
            "dtype": dtype,
            "is_contiguous": is_contiguous,
            "size": size,
            "strides": strides
        }
        if self.print_log:
            print('tensor_info:', tensor_info)

        return tensor_info

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
                elif actual_value.lower() == 'none':
                    return "Optional[]", None
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

    def _parse_arg(self, arg, aten_op_name, inplace_params, idx):
        parsed_arg = dict()
        # 解析参数名和值
        if ':' in arg:
            name, value = arg.split(':', 1)
            name = name.strip()
            value = value.strip()
        else:
            name = f"out{idx}" if self.out else f"arg_{idx}"
            value = arg.strip()

        # 检查是否为tensor（包括简化格式）
        if ('<' in value and '>' in value) and (('{' in value and '}' in value) or '%' in value or
                any(k in value for k in self.dtype_map)):
            tensor_info = self._parse_tensor_info(value, aten_op_name=aten_op_name, name=name, is_output=False)
            if tensor_info:
                # 检查这个参数是否是inplace的
                is_inplace = (
                        name in inplace_params or
                        f'idx_{idx}' in inplace_params or
                        self._is_inplace_arg_by_name(name, aten_op_name)
                )
                parsed_arg ={
                    "args_type": "Tensor",
                    "idx": idx,
                    "isWrite": is_inplace,
                    "name": name,
                    "value": tensor_info
                }
        else:
            # 根据值的格式推断类型
            if ':' not in value and '<' not in value and '>' not in value:
                value = name + ':' + value
                name = f"out{idx}" if self.out else f"arg_{idx}"
            args_type, parsed_value = self._infer_arg_type(value, param_name=name)
            parsed_arg = {
                    "args_type": args_type,
                    "idx": idx,
                    "isWrite": False,
                    "name": name,
                    "value": parsed_value
                }
        return parsed_arg

    def _parse_args(self, inputs_str, aten_op_name, inplace_params, idx=-1):
        # 分割参数 - 改进的解析逻辑
        if self.print_log:
            print('_parse_args:', inputs_str)

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
        for _i, arg in enumerate(args):
            if idx != -1:
                _i = idx
            if arg.startswith('tuple'):
                arg = arg[6:-1]
                parsed_args.append(self._parse_args(arg, aten_op_name, inplace_params, _i))
            elif arg.startswith('list'):
                arg = arg[5:-1]
                parsed_args.append(self._parse_args(arg, aten_op_name, inplace_params, _i))
            else:
                parsed_args.append(self._parse_arg(arg, aten_op_name, inplace_params, _i))

        return parsed_args

    def create_by_trace(self, inputs_str, aten_op_name=None, inplace_params=[]):
        # 如果是空字符串，返回空列表
        if not inputs_str.strip():
            return []
        self.print_log = aten_op_name in self.debug and 'parser' in self.debug[aten_op_name]
        if self.print_log:
            print('trace_args:', inputs_str)

        self.args = self._parse_args(inputs_str, aten_op_name, inplace_params)
        if self.print_log:
            print('parsed_args:', self.args)

    def get(self, mapping_model=None, mapping=None):
        if mapping_model is not None:
            _index = {}
            res = []
            if self.print_log:
                print('-'*20)
                print(self.args)
            if mapping is not None:
                for item in self.args:
                    for k in ['name', 'idx']:
                        if isinstance(item, list):
                            if self.print_log:
                                print(item, k, k in item[0])
                            if item[0][k] in mapping:
                                _index[mapping.index(item[0][k])] = item[0]['idx']
                        else:
                            if self.print_log:
                                print(item, k, k in item)
                            if item[k] in mapping:
                                _index[mapping.index(item[k])] = item['idx']
            else:
                mapping = self.args
                _index = {i:i for i in range(len(self.args))}
            if self.print_log:
                print('mapping:', mapping)
                print('_index:', _index)
            for k in range(len(mapping)):
                assert k in _index
                if 'args_type' not in self.args[_index[k]]:
                    this_arg = self.args[_index[k]][0]
                else:
                    this_arg = self.args[_index[k]]
                if this_arg['args_type'] not in ['Tensor', 'str']:
                    res.append('Scalar')
                    continue
                if 'size' not in this_arg['value']:
                    _value = this_arg['value']
                    if '*' in _value:
                        _scale, _value = _value.split('*')
                    else:
                        _scale = None
                    items = _value[1:-1].split('x')
                    this_arg['value'] = {}
                    if _scale:
                        this_arg['value']['scale'] = _scale
                    if self.dtype_map.get(items[-1], None) is not None:
                        this_arg['value']['dtype'] = self.dtype_map[items[-1]]
                        this_arg['value']['size'] = items[:-1]
                    else:
                        this_arg['value']['size'] = items
                if mapping_model == 'only_shape':
                    res.extend(this_arg['value']['size'])
                elif mapping_model == 'within_dtype':
                    res.append({'size': this_arg['value']['size'],
                              'dtype': this_arg['value']['dtype']})
                elif mapping_model == 'dte':
                    res.append({'size': this_arg['value']['size'],
                              'dtype': this_arg['value']['dtype'],
                              'is_contiguous': this_arg['value']['is_contiguous']})
                else:
                    res.append(this_arg['value'])
            return res
        return self.args