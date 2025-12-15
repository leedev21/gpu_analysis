import argparse
import numpy as np
from tqdm import tqdm
from theory.core.utils import each_file, read_file, read_json
from theory.core.backend.hardware import hw_loader
from theory.core.collections.op.performance import op_object
import math

hw_set = set()

def op_efficiency(hw: str,
        op_name: str,
        shape: list,
        duration: float,
        decoding: bool=False,
        causal: bool=False,
        hw_cfg: dict=None,
        upper_bound: bool=True):
    backend = hw_loader.get(hw)
    if hw_cfg:
        for attr in hw_cfg:
            for key in hw_cfg[attr]:
                backend.config(attr, key, hw_cfg[attr][key])
    backend.config('efficiency', 'upper_bound', upper_bound)
    return op_object(op_name, shape, backend, duration, decoding=decoding, causal=causal)


def run_op(hw: str,
        op_name: str,
        shape: list,
        decoding: bool=False,
        causal: bool=False,
        hw_cfg: dict=None,
        upper_bound: bool=True):
    backend = hw_loader.get(hw)
    if hw_cfg:
        for attr in hw_cfg:
            for key in hw_cfg[attr]:
                backend.config(attr, key, hw_cfg[attr][key])
    backend.config('efficiency', 'upper_bound', upper_bound)
    return op_object(op_name, shape, backend, decoding=decoding, causal=causal)


class OpExecutor:
    """Class for executing and analyzing operations with hardware efficiency calculations."""

    def run_op_calibration(self, method, case, input, recompute=True):
        excluded_terms = ['flops', 'io', 'efficient', 'latency', 'op_type', 'params', 'activation', 'utilization', 'precision']
        self.data = {}
        for hw, op_name, input_shape, duration_all, dtype, kwargs in self.load_multi_op_readout(case, input, excluded_terms):
            self.calibration_list = []
            hw_set.add(hw)
            duration = duration_all[0] if duration_all else None
            item = '_'.join([hw, op_name, dtype]).replace('::','.')
            if item not in self.data:
                self.data[item] = {
                    'op_name': op_name.replace('::','.'),
                    'hardware': hw,
                    'dtype': dtype,
                    'shape': [],
                    'select_group': [],
                    'x_data': [],
                    'efficiency': [],
                    'duration': [],
                    'theory_upper': [],
                    'gap_upper': [],
                    'theory_calib': [],
                    'gap_calib': [],
                    'efficiency_popt': [],
                    'efficiency_calib': [],
                    'efficiency_verify': [],
                }
            if recompute and duration:
                theory_efficiency = op_efficiency(hw, op_name, input_shape, duration)
            if case == 'IO':
                self.calibration_list.append('L3_bandwidth')
                self.data[item]['select_group'] = ['io']
                self.data[item]['x_data'].append(round(math.log(kwargs['io']), 2))
                self.data[item]['efficiency'].append(round(theory_efficiency['io_utilization']*100 if recompute else eval(kwargs['io_efficiency'][:-1]), 2))
                self.data[item]['duration'].append(round(duration, 2))
            elif case == '2D':
                self.calibration_list.append('2D_fp16')
                self.data[item]['select_group'] = ['flops_2D']
                self.data[item]['x_data'].append(round(math.log(kwargs['flops']), 2))
                self.data[item]['efficiency'].append(round(eval(kwargs['flops_efficiency'][:-1]), 2))
                self.data[item]['duration'].append(round(duration, 2))
            elif case == '1D':
                self.calibration_list.append('1D')
                if 'flops' not in kwargs:
                    kwargs['flops'] = theory_efficiency['flops_utilization']['sfu_scale']*kwargs['flops_SFU'] + kwargs['flops_1D']
                    flops_efficiency = eval(kwargs['flops_efficiency_1D'][:-1]) + eval(kwargs['flops_efficiency_SFU'][:-1])
                else:
                    flops_efficiency = eval(kwargs['flops_efficiency'][:-1])
                self.data[item]['select_group'] = ['flops_1D']
                self.data[item]['x_data'].append(round(math.log(kwargs['flops']), 2))
                self.data[item]['efficiency'].append(round(flops_efficiency, 2))
                self.data[item]['duration'].append(round(duration, 2))
            elif case == '2D_fp8':
                self.calibration_list.append('2D_fp8')
                self.data[item]['select_group'] = ['flops_2D']
                self.data[item]['x_data'].append(round(math.log(kwargs['flops']), 2))
                self.data[item]['efficiency'].append(round(eval(kwargs['flops_efficiency'][:-1]), 2))
                self.data[item]['duration'].append(round(duration, 2))
            elif case == 'Comm':
                self.calibration_list.append(op_name.replace('c10d::', ''))
                self.data[item]['select_group'] = ['io']
                io_efficiency = eval(kwargs['io_efficiency'][:-1])
                if not duration:
                    theory_latency = run_op(hw, op_name, input_shape, hw_cfg={'efficiency': {'D-D': io_efficiency}}, upper_bound=True)
                    duration = theory_latency['latency']
                    kwargs['io'] = theory_latency['io']
                self.data[item]['x_data'].append(round(math.log(kwargs['io']), 2))
                self.data[item]['efficiency'].append(round(io_efficiency, 2))
                self.data[item]['duration'].append(round(duration, 2))
            elif case == 'FA':
                self.data[item]['select_group'] = ['flops_2D', 'flops_1D', 'io']
                for var in self.data[item]['select_group']:
                    if var == 'io':
                        self.calibration_list.append('L3_bandwidth')
                        self.data[item]['x_data'].append(round(math.log(theory_efficiency['io']), 2))
                        self.data[item]['efficiency'].append(round(theory_efficiency['io_utilization']*100, 2))
                    elif var == 'flops_1D':
                        self.calibration_list.append('1D')
                        self.data[item]['x_data'].append(round(math.log(theory_efficiency['flops_utilization']['sfu_scale'] *
                            sum(theory_efficiency['flops']['SFU']) + sum(theory_efficiency['flops']['1D'])), 2))
                        self.data[item]['efficiency'].append(round((theory_efficiency['flops_utilization']['1D'] + \
                            theory_efficiency['flops_utilization']['SFU'])*100, 2))
                    elif var == 'flops_2D':
                        self.calibration_list.append('2D_fp16')
                        self.data[item]['x_data'].append(round(math.log(sum(theory_efficiency['flops']['2D'])), 2))
                        self.data[item]['efficiency'].append(round(theory_efficiency['flops_utilization']['2D']*100, 2))
                self.data[item]['duration'].append(round(duration, 2))

            else:
                print('calibration of this case not support Now!', case, item)
                exit()

            theory_latency = run_op(hw, op_name, input_shape, upper_bound=True)
            diff = round(abs(theory_latency['latency'] - duration) / duration * 100, 2)
            self.data[item]['theory_upper'].append(round(theory_latency['latency'], 2))
            self.data[item]['gap_upper'].append(diff)

            theory_latency = run_op(hw, op_name, input_shape, upper_bound=False)
            diff = round(abs(theory_latency['latency'] - duration) / duration * 100, 2)
            self.data[item]['theory_calib'].append(round(theory_latency['latency'], 2))
            self.data[item]['gap_calib'].append(diff)
            self.data[item]['shape'].append(input_shape)

        for case_id in self.data:
            backend = hw_loader.get(self.data[case_id]['hardware'])
            # Number of variables per group
            M = len(self.data[case_id]['select_group'])
            total = len(self.data[case_id]['x_data'])
            N = total // M  # Number of groups
            # 1. Reshape multi-value fields into N×M arrays
            x_data = np.array(self.data[case_id]['x_data']).reshape(M, N)
            efficiency = np.array(self.data[case_id]['efficiency']).reshape(M, N)

            # 2. Sort only the first group and apply to all fields
            indices = np.argsort(x_data[0])
            x_data[0] = x_data[0][indices]
            efficiency[0] = efficiency[0][indices]

            # Sort all single-value fields using the same indices
            fields_to_sort = ['duration', 'theory_upper', 'gap_upper', 'theory_calib', 'gap_calib']
            for field in fields_to_sort:
                self.data[case_id][field] = np.array(self.data[case_id][field])[indices]

            # 3. Fit each group
            self.data[case_id]['efficiency_popt'] = []
            self.data[case_id]['efficiency_calib'] = []
            self.data[case_id]['efficiency_verify'] = []

            for i in range(M):
                popt, draw_calib = backend.calibration(method, case, x_data[i], efficiency[i], self.calibration_list[i])
                _, draw_verify = backend.verification(method, case, x_data[i], efficiency[i], self.calibration_list[i])
                self.data[case_id]['efficiency_popt'].append(popt)
                self.data[case_id]['efficiency_calib'].append(draw_calib)
                self.data[case_id]['efficiency_verify'].append(draw_verify)

            # 4. Update the data structure with sorted arrays
            self.data[case_id]['x_data'] = x_data.tolist()
            self.data[case_id]['efficiency'] = efficiency.tolist()

        self.data = self.convert_numpy_to_lists(self.data)

        return self.data

    def convert_numpy_to_lists(self, data):
        """
        递归地将所有NumPy数组转换为Python列表。
        Args:
            data: 任何数据结构, 可能包含NumPy数组
        Returns:
            转换后的数据结构, 所有NumPy数组都被转换为Python列表
        """
        if isinstance(data, np.ndarray):
            return [round(float(x), 2) if isinstance(x, (float, np.float32, np.float64)) else self.convert_numpy_to_lists(x) for x in data]
        elif isinstance(data, list):
            return [self.convert_numpy_to_lists(x) for x in data]
        elif isinstance(data, dict):
            return {k: self.convert_numpy_to_lists(v) for k, v in data.items()}
        elif isinstance(data, (np.float32, np.float64)):
            return round(float(data), 2)
        elif isinstance(data, (np.int32, np.int64)):
            return int(data)
        else:
            return data

    def load_multi_op_readout(self, case, input, excluded_terms):
        op_filters = {
            'Launch': ['aten::contigous', 'aten::slice'],
            'IO': ['aten::add', 'aten::silu', 'aten::sub', 'aten::mul', 'aten::fill_', 'aten::copy_', 'aten::gelu'],
            '1D': ['apex::fused_layer_norm_cuda.forward', 'apex::fused_layer_norm_cuda.rms_forward', 'apex::fused_rotary_positional_embedding.backward', 'apex::fused_rotary_positional_embedding.forward'],
            '2D': ['aten::mm', 'aten::baddbmm', 'aten::bmm'],
            'Comm': ['c10d::all2all', 'c10d::allreduce', 'c10d::allgather', 'c10d::reduce_scatter'],
            'FA': ['flash_attn::flash_attn_qkvpacked_func', 'flash_attn::flash_attn_func', 'flash_attn::flash_attn_interface.flash_attn_func', 'aten::scaled_dot_product_attention'],
        }
        for log_path, log_file in tqdm(list(each_file(input, '.log', True)), desc="Processing log files", unit="file"):
            hw = log_file.split('/')[-1].split('_')[0]
            # print("log_file",log_file)
            special_nameplace = ""
            if '[small]_' in log_file:
                special_nameplace = '[small]_'
            elif '[long]_' in log_file:
                special_nameplace = '[long]_'
            op_name = log_file.split('_2025')[0].split(f'{hw}_')[1]
            op_name = op_name.replace('[small]_', '').replace('[long]_', '')
            op_name = op_name.replace('__', '::').replace('-', '::')
            data = read_file('', log_file)
            schema = []
            formatted_data_solo = []
            if op_name not in op_filters[case]:
                continue
            if hw == 'A100SXM':
                hw += '40G'       # 当前CSE仅有40G版本

            for line in data:
                line = line.strip().split('|')
                if len(line) == 1:
                    continue
                line = line[1:-1]
                if not schema:
                    schema = list(map(lambda x: x.strip(), line))
                    custom_fields = self.extract_custom_fields(schema, excluded_terms)
                    # print(list(schema))
                    # print(custom_fields)
                else:
                    items = list(map(lambda x: x[1].strip() if schema[x[0]] in ['dtype', 'precision'] or any(k in x[1] for k in['%', '...', 'N/A'])  or x[1].strip() == '' else eval(x[1].strip()), enumerate(line)))
                    # print(list(items))
                    input_shape = self.normalize_input_shape(items[schema.index('input_shape')])
                    duration_all = [items[schema.index('duration_min(us)')],  items[schema.index('duration_mean(us)')], items[schema.index('duration_max(us)')]] if 'duration_min(us)' in schema else None
                    dtype = items[schema.index('dtype')]
                    kwargs = {}
                    for idx, field_name in custom_fields.items():
                        kwargs[field_name] = items[idx]
                    yield hw, op_name, input_shape, duration_all, dtype, kwargs

    def normalize_input_shape(self, shape):
        # If already in double brackets format, return as is
        if isinstance(shape, list) and len(shape) > 0 and isinstance(shape[0], list):
            return shape

        # If it's a single-level list, wrap it in another list
        if isinstance(shape, list):
            return [shape]

        # If it's something else, wrap it in double lists
        return [[shape]]

    def extract_custom_fields(self, schema, excluded_terms):
        """
        Extract custom fields from schema by excluding standard parameters
        and fields containing specific keywords.

        Args:
            schema (list): List of field names from the schema

        Returns:
            dict: Dictionary mapping field indices to field names for custom fields
        """
        # Combined exclusion criteria

        custom_fields = {}
        # print("Processing schema:", schema)
        for idx, field in enumerate(schema):
            field_lower = field

            # Check if field should be excluded
            should_exclude = False
            for term in excluded_terms:
                if term in field_lower:
                    should_exclude = True
                    # print(f"Excluding field '{field}' because it contains '{term}'")
                    break

            if should_exclude:
                custom_fields[idx] = field
                # print(f"Including field '{field}' as custom field")
        # print("Final custom fields:", custom_fields)
        return custom_fields

    def get_hardware_info(self):
        """
        Get hardware specific information.

        Returns:
            dict: Hardware information including launch time and max efficiency for each hardware
        """
        hardware_info = {}
        for hardware_name in hw_set:
            backend = hw_loader.get(hardware_name)
            if not backend:
                continue
            # Get efficiency_max from specified calibration's asymmetric_sigmoid first element
            efficiency_max = {}
            for calibration in self.calibration_list:
                try:
                    if calibration in backend._efficiency.popt and 'asymmetric_sigmoid' in backend._efficiency.popt[calibration]:
                        efficiency_max[calibration] = backend._efficiency.popt[calibration]['asymmetric_sigmoid'][0]
                    else:
                        efficiency_max[calibration] = 0.0
                except (KeyError, IndexError):
                    efficiency_max[calibration] = 0.0

            # Get launch_time from launch std first element
            try:
                if 'launch' in backend._efficiency.popt and 'std' in backend._efficiency.popt['launch']:
                    launch_time = backend._efficiency.popt['launch']['std'][0]
                else:
                    launch_time = 0.0
            except (KeyError, IndexError):
                launch_time = 0.0

            hardware_info[hardware_name] = {
                'launch_time': launch_time,
                'efficiency_max': efficiency_max
            }

        return hardware_info

    def get_method_list(self):
        """
        Get list of available curve fitting methods.
        Default method is asymmetric_sigmoid.

        Returns:
            list: List of available fitting methods
        """
        method = [
            'asymmetric_sigmoid',  # 默认拟合方法
            'sigmoid',            # 标准sigmoid函数
            'polynomial',         # 多项式拟合
            'exponential',        # 指数拟合
            'logarithmic',        # 对数拟合
            'linear'             # 线性拟合
        ]
        return method

    def select_data_group(self, group_name):
        """
        Get operators for specified group.

        Args:
            group_name (str): Name of the operator group
            dtype (str, optional): Data type. Defaults to None.

        Returns:
            list: List of operators in the group
        """
        op_group = {
            'IO': ['aten.add', 'aten.silu', 'aten.sub', 'aten.mul', 'aten.fill_', 'aten.copy_', 'aten.gelu'],
            '1D': ['apex.fused_layer_norm_cuda.forward', 'apex.fused_layer_norm_cuda.rms_forward', 'apex.fused_rotary_positional_embedding.backward', 'apex.fused_rotary_positional_embedding.forward'],
            '2D': ['aten.mm', 'aten.baddbmm', 'aten.bmm'],
            'Comm': ['c10d.all2all', 'c10d.allreduce', 'c10d.allgather', 'c10d.reduce_scatter'],
            'FA': ['flash_attn.flash_attn_qkvpacked_func', 'flash_attn.flash_attn_func', 'flash_attn.flash_attn_interface.flash_attn_func', 'aten.scaled_dot_product_attention']
        }
        return op_group.get(group_name, [])

    def get_plot_params(self):
        show_name = {
            # Gap Limits
            'max_gap_limit': '最大差异限制',
            'min_gap_limit': '最小差异限制',

            # X Axis Logarithmic Values
            'x_data_io': 'log(IO)',
            'x_data_flops_1D': 'log(flops_1D)',

            # Y Axis Efficiencies
            'efficiency_io': '带宽利用率',
            'efficiency_flops_1D': '1D算力利用率',
            'efficiency_flops_2D': '2D算力利用率',

            # Y Axis Calibration Efficiencies
            'efficiency_calib_io': '带宽利用率拟合曲线',
            'efficiency_calib_flops_1D': '1D算力利用率拟合曲线',
            'efficiency_calib_flops_2D': '2D算力利用率拟合曲线',

            # Y Axis Duration Calibration
            'efficiency_verify_io': '带宽利用率校准曲线',
            'efficiency_verify_flops_1D': '1D算力利用率校准曲线',
            'efficiency_verify_flops_2D': '2D算力利用率校准曲线',
        }

        return show_name