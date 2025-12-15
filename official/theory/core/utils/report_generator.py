from collections import OrderedDict, defaultdict
from typing import Optional, Dict, List, Tuple
import os
import datetime

global_column_order = []

def format_data(raw_data, with_memory=False):
    """Format raw data into standardized structure while preserving dynamic fields.

    Args:
        raw_data: Output from data_gather() (any structure)

    Returns:
        Formatted data dictionary that:
        - Preserves all original fields
        - Applies scientific notation to large numbers
        - Converts utilizations to percentages
        - Maintains original structure of dynamic fields
    """

    def format_efficiency_data(efficiency, with_memory):
        """
        Format data dictionary with scientific notation for numeric values,
        and percentage format for utilization values.

        Args:
            efficiency (dict): Input dictionary with values to format

        Returns:
            dict: Dictionary with formatted values
        """
        result = {}
        if not efficiency:
            return {
                'io': 0,
                'io_efficiency': 0
            }

        for key, value in efficiency.items():
            if value is None:
                result[key] = 0
                continue

            if 'utilization' in key:
                # Format as percentage with % symbol
                if isinstance(value, dict):
                    for cls in value:
                        if cls == 'sfu_scale':
                            continue
                        item = key.replace('utilization', 'efficiency') + f'_{cls}'
                        if isinstance(value[cls], str) and '%' in value[cls]:
                            result[item] = value[cls]
                        else:
                            result[item] = f"{value[cls] * 100:.2f}%"
                else:
                    item = key.replace('utilization', 'efficiency')
                    if isinstance(value, str) and '%' in value:
                        result[item] = value
                    else:
                        result[item] = f"{value * 100:.2f}%"
            elif with_memory and key in ['params', 'activation']:
                # Convert to scientific notation string then back to float
                result[key] = f"{value:.2e}"
            elif key in ['io', 'flops'] or key.startswith('flops') and not key.startswith('device_tflops'):
                # Convert to scientific notation string then back to float
                if isinstance(value, dict):
                    for cls in value:
                        if isinstance(value[cls], list):
                            result[f"{key}_{cls}"] = f"{sum(value[cls]):.2e}"
                        else:
                            result[f"{key}_{cls}"] = f"{value[cls]:.2e}"
                else:
                    result[key] = f"{value:.2e}"
            elif key == 'device_tflops':
                tflops_value = value/1000000000000
                result[key] = f"{tflops_value:.2e} TFLOPS"
            # else:theory/core/utils/report_generator.py
            #     # Keep other values unchanged
            #     result[key] = value

        return result

    def build_ordered_report(formatted_data):
        """构建严格按顺序的无前缀有序字典"""
        ordered = OrderedDict()
        if 'kwargs1' in formatted_data:
            kwargs1 = formatted_data['kwargs1']
            ordered.update(OrderedDict(
                (k, v) for k, v in kwargs1.items()
                if k not in ordered
            ))

        # 1.1 添加 config 参数（排除 dtype/precision）
        if 'config' in formatted_data:
            config = formatted_data['config'].copy()

            # 根据原始字段名决定列名
            if 'dtype' in config:
                dtype_value = config.pop('dtype', 'N/A')
                precision_column_name = 'dtype'
            elif 'precision' in config:
                dtype_value = config.pop('precision', 'N/A')
                precision_column_name = 'precision'
            else:
                dtype_value = 'N/A'
                precision_column_name = 'precision'  # 默认使用precision

            ordered.update(config)  # 添加 hardware 等字段

        # 1.2 name
        if 'name' in formatted_data:
            ordered['name'] = formatted_data['name']

        # 2.1 shape
        if 'shape' in formatted_data:
            test_shape = formatted_data['shape']
            ordered.update(OrderedDict(
                (k, v) for k, v in test_shape.items()
                if k not in ordered
            ))

        # 2.2 kwargs
        if 'kwargs' in formatted_data:
            test_case = formatted_data['kwargs']
            ordered.update(OrderedDict(
                (k, v) for k, v in test_case.items()
                if k not in ordered
            ))

        # 3.1 插入 dtype 或 precision（根据原始数据决定）
        if 'config' in formatted_data:
            ordered[precision_column_name] = dtype_value

        # 3.2 Duration参数
        if 'duration' in formatted_data:
            duration = formatted_data['duration']
            ordered.update(OrderedDict(
                (k, v) for k, v in duration.items()
                if k not in ordered  # 防止覆盖config
            ))

        # 4. Theory参数
        if 'theory' in formatted_data:
            theory = formatted_data['theory']
            ordered.update(theory.items())

        # 5. Memory parameters
        if 'memory' in formatted_data:
            memory = formatted_data['memory']
            ordered.update(memory.items())

        # 6. Performance parameters
        if 'performance' in formatted_data:
            performance = formatted_data['performance']
            ordered.update(performance.items())

        # 7. Process Efficiency section
        if 'efficiency' in formatted_data:
            efficiency = formatted_data['efficiency']

            # 7.1 params & activation (exact match)
            if 'params' in efficiency:
                ordered[efficiency.get('params_key', 'params')] = efficiency.get('params', 'N/A')
            if 'activation' in efficiency:
                ordered[efficiency.get('activation_key', 'activation')] = efficiency.get('activation', 'N/A')

            # 7.2 All flops fields (excluding utilization)
            flops_keys = [k for k in efficiency if 'flops' in k.lower() and 'efficiency' not in k.lower()]
            for k in flops_keys:
                ordered[k] = efficiency[k]

            # 7.3 All io fields (excluding utilization)
            io_keys = [k for k in efficiency if 'io' in k.lower() and 'efficiency' not in k.lower()]
            for k in io_keys:
                ordered[k] = efficiency[k]

            # 7.4 flops_utilization fields
            flops_util_keys = [k for k in efficiency if 'flops' in k.lower() and 'efficiency' in k.lower()]
            for k in flops_util_keys:
                ordered[k] = efficiency[k]

            # 7.5 io_utilization fields
            io_util_keys = [k for k in efficiency if 'io' in k.lower() and 'efficiency' in k.lower()]
            for k in io_util_keys:
                ordered[k] = efficiency[k]

        if 'compare' in formatted_data:
            for k in formatted_data['compare']:
                ordered[k] = formatted_data['compare'][k]

        if 'acc' in formatted_data:
            for k in formatted_data['acc']:
                ordered[k] = formatted_data['acc'][k]
        return ordered

    def update_global_order(formatted_data: OrderedDict):
        """
        从单个 formatted_data 提取字段顺序，更新全局顺序
        规则：
        1. 保留全局列表中的所有已有字段及其顺序
        2. 将新字段按照它们在当前数据中的相对顺序插入到适当位置
        """
        global global_column_order

        # 如果全局列表为空，直接使用当前数据的顺序
        if not global_column_order:
            global_column_order.extend(formatted_data.keys())
            # print("global_column_order (initialized)", global_column_order)
            return

        # 找出当前数据中的新字段
        new_fields = [key for key in formatted_data.keys() if key not in global_column_order]

        if not new_fields:
            # print("global_column_order (unchanged)", global_column_order)
            return

        # 为每个新字段找到合适的插入位置
        current_fields = list(formatted_data.keys())

        # 处理每个新字段
        for new_field in new_fields:
            # 找到新字段在当前数据中的位置
            new_field_pos = current_fields.index(new_field)

            # 找到新字段前后的已存在于全局顺序中的字段
            before_fields = []
            after_fields = []

            # 检查当前数据中，新字段之前的字段
            for field in current_fields[:new_field_pos]:
                if field in global_column_order:
                    before_fields.append(field)

            # 检查当前数据中，新字段之后的字段
            for field in current_fields[new_field_pos+1:]:
                if field in global_column_order:
                    after_fields.append(field)

            # 确定插入位置
            if before_fields:
                # 在最后一个前置字段之后插入
                last_before_field = before_fields[-1]
                insert_pos = global_column_order.index(last_before_field) + 1
            elif after_fields:
                # 在第一个后置字段之前插入
                first_after_field = after_fields[0]
                insert_pos = global_column_order.index(first_after_field)
            else:
                # 没有参考字段，添加到末尾
                insert_pos = len(global_column_order)

            # 插入新字段
            global_column_order.insert(insert_pos, new_field)


    formatted = raw_data.copy()

    # Keep original field names (dtype or precision), don't unify to dtype
    config_data = {}
    if 'dtype' in raw_data['config']:
        config_data['dtype'] = raw_data['config']['dtype']
    elif 'precision' in raw_data['config']:
        config_data['precision'] = raw_data['config']['precision']

    if 'hardware' in raw_data['config']:
        config_data['hardware'] = raw_data['config']['hardware']

    if 'test_id' in raw_data['config']:
        config_data['test_id'] = raw_data['config']['test_id']

    formatted['config'] = config_data

    # Format duration data
    if 'duration' in raw_data:
        formatted['duration'] = {}
        if isinstance(raw_data['duration'], dict):
            for k in ['mean', 'min', 'max']:
                formatted['duration'][f"duration_{k}(us)"] = round(raw_data['duration'][k], 2)
        elif isinstance(raw_data['duration'], list):
            assert len(raw_data['duration']) == 3, f"duration列表长度必须为3, 当前长度为{len(raw_data['duration'])}"
            formatted['duration'][f"duration_min(us)"] = min(raw_data['duration'])
            formatted['duration'][f"duration_mean(us)"] = sorted(raw_data['duration'])[1]
            formatted['duration'][f"duration_max(us)"] = max(raw_data['duration'])
        else:
            formatted['duration'][f"duration(us)"] = round(raw_data['duration'], 2)

    # Format test case data (preserve structure)
    if 'shape' in raw_data:
        formatted['shape']  = {k: v for k, v in raw_data['shape'].copy().items() if k != 'name'}

    # Format efficiency data dynamically
    if 'efficiency' in raw_data:
        efficiency = raw_data['efficiency']
        # format_efficiency_data
        formatted['efficiency'] = format_efficiency_data(efficiency, with_memory)

    # Format theory data
    if 'theory' in raw_data:
        theory = raw_data['theory']
        formatted['theory'] = {
            'theory_latency(us)': round(theory.get('latency', 0), 2) if theory else 0,
        }

    # Format memory data
    if 'memory' in raw_data:
        memory = raw_data['memory']
        formatted['memory'] = {}
        for key, value in memory.items():
            if isinstance(value, (int, float)):
                # Convert to appropriate memory units (MB/GB)
                if value >= 1024 * 1024 * 1024:  # >= 1GB
                    formatted['memory'][key] = f"{value / (1024 * 1024 * 1024):.2f}GB"
                elif value >= 1024 * 1024:  # >= 1MB
                    formatted['memory'][key] = f"{value / (1024 * 1024):.2f}MB"
                elif value >= 1024:  # >= 1KB
                    formatted['memory'][key] = f"{value / 1024:.2f}KB"
                else:
                    formatted['memory'][key] = f"{value:.2f}B"
            else:
                formatted['memory'][key] = value

    # Format performance data
    if 'performance' in raw_data:
        performance = raw_data['performance']
        formatted['performance'] = {}
        for key, value in performance.items():
            if isinstance(value, (int, float)):
                formatted['performance'][key] = round(value, 2)
            else:
                formatted['performance'][key] = value


    ordered_formatd_report = build_ordered_report(formatted)
    update_global_order(ordered_formatd_report)
    format_data._columns = global_column_order

    return ordered_formatd_report


class ReportGenerator:
    def __init__(self, name: str):
        self.name = name
        self.op_first_timestamps = {}  # 格式: {op_name: timestamp}
        self.entry = {
            'log_cache': OrderedDict(),
            'column_widths': OrderedDict(),
            'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        }

    def DataFrame(self, formatted_data: dict):
        self._merge_data(formatted_data)

    def to_report(self, filename: str, index=None) -> str:
        """生成日志文件"""
        # 1. 验证基础文件名
        if not filename.strip():
            raise ValueError("Base filename cannot be empty or whitespace")

        # 2. 处理目录路径
        dir_path = os.path.dirname(filename)
        if not dir_path:
            dir_path = os.getcwd()
            filename = os.path.join(dir_path, os.path.basename(filename))

        # 3. 创建目录
        os.makedirs(dir_path, exist_ok=True)

        # 4. 生成表格内容
        table = self._format_table()  # 调用表格生成方法

        # 5. 写入文件内容
        with open(filename, 'w') as f:
            # 写入文件头
            f.write(f"Test Case: {self.name}\n")
            # 写入表格内容
            f.write(table + "\n")

        return os.path.abspath(filename)

    def to_csv(self, filename: str, index=False) -> str:
        """
        生成CSV格式的报告
        :param filename: 基础文件名（无需后缀，例如 "Conv2d_perf"）
        :param index: 是否保留索引列（默认False）
        :return: 生成的完整文件路径
        """
        pass

    def _merge_data(self, new_data: dict):
        log_cache = self.entry['log_cache']
        current_length = 0

        if log_cache:
            first_key = next(iter(log_cache))
            current_length = len(log_cache[first_key])

        # 处理现有键：添加新值或N/A
        for key in list(log_cache.keys()):
            if key in new_data:
                log_cache[key].append(new_data[key])
            else:
                log_cache[key].append('N/A')

        # 处理新数据中的新键
        for key, value in new_data.items():
            if key not in log_cache:
                # 填充当前长度的N/A，然后添加当前值
                log_cache[key] = ['N/A'] * current_length
                log_cache[key].append(value)

    def _calculate_widths(self):
        self.entry['column_widths'].clear()
        for field, values in self.entry['log_cache'].items():
            content_width = max(len(str(v)) for v in values) if values else 0
            self.entry['column_widths'][field] = max(len(field), content_width) + 2

    def reindex(self, columns=None):
        """
        按指定的列顺序重新排序log_cache中的数据

        Args:
            columns: 可选的列顺序列表。如果未提供，则使用全局列顺序
        """
        # 优先使用传入的列顺序，如果没有则使用全局顺序
        column_order = columns if columns is not None else global_column_order

        if not column_order or not self.entry['log_cache']:
            return  # 没有列顺序或日志缓存为空，不需要操作

        # 创建一个新的有序字典来按指定顺序重新组织数据
        reordered_cache = OrderedDict()

        # 首先按照指定的列顺序添加已有的列
        for col in column_order:
            if col in self.entry['log_cache']:
                reordered_cache[col] = self.entry['log_cache'][col]

        # 然后添加任何指定顺序中未包含但在log_cache中存在的列
        # 这是为了确保不会丢失任何数据
        for col in self.entry['log_cache']:
            if col not in reordered_cache:
                reordered_cache[col] = self.entry['log_cache'][col]

        # 用重新排序的缓存替换原来的缓存
        self.entry['log_cache'] = reordered_cache


    def _format_table(self) -> str:
        self._calculate_widths()
        col_widths = self.entry['column_widths']
        header = "|".join(f" {k:^{w-2}} " for k, w in col_widths.items())
        header = f"|{header}|"

        rows = []

        num_rows = len(next(iter(self.entry['log_cache'].values())))

        for i in range(num_rows):
            row = "|".join(
                f" {str(values[i]):^{col_widths[key]-2}} "
                for key, values in self.entry['log_cache'].items()
            )
            rows.append(f"|{row}|")

        return "\n".join([header] + rows)