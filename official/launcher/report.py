import re
import json
import os
import datetime
from theory.run import run_op, op_efficiency, ReportGenerator, format_data
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
from omegaconf import OmegaConf
from collections import OrderedDict, defaultdict
from typing import Optional, Dict, List, Tuple
from copy import deepcopy


class Report(object):
    def __init__(self):
        self.op_log_files = {}
        self.generators = {}
        self.last_reported_index = -1
        self.data = []
        self.theory_latency = []

    def __call__(self, test_case, _mean, _min, _max, acc):
        self.data.append((test_case, _mean, _min, _max, acc))

    def add_target(self, target):
        self.target = target

    def config(self, attr, value):
        if hasattr(self, attr):
            setattr(self, attr, value)
        else:
            self.conf[attr] = value

    def get_shape(self, test_case):
        _in = []
        if hasattr(test_case, 'input'):
            for tensor in test_case.input:
                if isinstance(tensor, ListConfig):
                    _in.append(list(tensor))
                elif isinstance(tensor, DictConfig):
                    if '_tensor' in tensor:
                        _in.append(list(tensor['_tensor']))
                # Skip scalar parameters like epsilon, they are not tensors
                # and should not be included in shape information
        return _in

    def check_limit(self, test_case, config, limit):
        shape = self.get_shape(test_case)
        if shape:
            theory_latency = run_op(config['hw'], test_case.name, shape)
            if limit and theory_latency['latency'] >= limit:
                print('limit:', limit, '; theory:', theory_latency['latency'])
                return True
            self.theory_latency.append(theory_latency)
        return False

    def report(self, args, config):
        print('='*50, 'Report', '='*50)
        print(args)

        op_data = {}
        for i, line in enumerate(self.data):
            test_case, _mean, _min, _max, acc = line
            shape = self.get_shape(test_case)
            efficient = 0
            theory_latency = 0
            # try:
            efficient = op_efficiency(config['hw'], test_case.name, shape, _min)
            theory_latency = self.theory_latency[i]
            # except:
            #     pass
            _mean = round(_mean, 2)
            _min = round(_min, 2)
            _max = round(_max, 2)
            print(test_case, _mean, _min, _max, theory_latency['latency'], efficient['flops_utilization'], efficient['io_utilization'])

            op_name = self.extract_operation_name(test_case)
            print("test_case", test_case)
            processed_data = {
                'test_case': test_case,
                'mean': _mean,
                'min': _min,
                'max': _max,
                'efficient': efficient,
                'theory_latency': theory_latency,
                'shape': shape
            }

            # 检查键是否存在，如果不存在则初始化为空列表
            if op_name not in op_data:
                op_data[op_name] = []
            op_data[op_name].append(processed_data)
        hw_name = config.get('hw', 'Unknown')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Create a separate log file for each operation
        log_files = {}
        for op_name, data in op_data.items():
            log_path = self.save_log_file(op_name, hw_name, timestamp, data, args, config)
            log_files[op_name] = log_path

        return log_files

    def instant_report(self, args, config):
        if not hasattr(self, 'last_reported_index'):
            self.last_reported_index = -1
            self.op_log_files = {}
            self.first_file_created = {}

        if len(self.data) <= self.last_reported_index + 1:
            return None

        i = self.last_reported_index + 1
        test_case, _mean, _min, _max, acc = self.data[i]
        shape = self.get_shape(test_case)
        hw_name = config.get('hw', 'Unknown')
        if not hasattr(test_case, 'name'):
            return
        efficient = None
        theory_latency = None
        if not acc:
            try:
                efficient = op_efficiency(hw_name, test_case.name, shape, _min)
                theory_latency = self.theory_latency[i] if i < len(self.theory_latency) else None
                # print("efficient",efficient)
                # print("theory_latency",theory_latency)
            except Exception as e:
                print(f"Error calculating efficiency: {e}")

            # 生成格式化数据
        raw_datas = data_gather(args, config, self.data[i], efficient, theory_latency)

        op_name = raw_datas[0]['config']['op_name']
        # 关键逻辑：生成或复用文件路径
        if op_name not in self.op_log_files:
            # 首次生成时间戳和文件名
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_report = f"{args.training.report}/{hw_name}_{op_name}_{timestamp}.log"
            self.op_log_files[op_name] = model_report
            print(f"📁 新日志文件已创建：{model_report}")  # 新增提示
        else:
            # 复用已有文件路径
            model_report = self.op_log_files[op_name]
            print(f"📂 复用已有日志文件：{model_report}")  # 新增提示
         # 获取或创建生成器

        name = f"{hw_name}_{op_name.replace('::', '-')}"
        if op_name not in self.generators:
            self.generators[op_name] = ReportGenerator(name=name)
        generator = self.generators[op_name]

        # 覆盖写入文件
        for raw_data in raw_datas:
            formatted_data = format_data(raw_data)
            generator.DataFrame(formatted_data)
        generator.reindex(columns=format_data._columns)
        generator.to_report(model_report, index=False)

        # 更新索引
        self.last_reported_index = i
        return model_report

def data_gather(args, config, test_case_data, efficient, theory):
    """Extract and structure raw performance test data dynamically.

    Args:
        args: Command line arguments dict (any structure)
        hw: Hardware name string
        test_case_data: Tuple of (test_case_dict, mean_time, min_time, max_time)
        efficient: Efficiency metrics dict (any structure)
        theory: Theoretical latency dict (any structure)

    Returns:
        Dictionary containing all input data in structured form, preserving
        original nested structures while adding some standard fields.
    """
    test_case, _mean, _min, _max, acc = test_case_data
    op_name = test_case.get('name', 'unknown').replace('::', '-')
    hw = config.get('hw', 'Unknown')
    # Create structured raw data with flat organization
    data = []
    result = {
        'config': {
            'hardware': hw,
            'precision': 'unknown',
            'op_name': op_name
        },
        'test_case': {
            'name': test_case.get('name'),
            **{k: v for k, v in test_case.items() if k != 'name'}
        },
        'raw_args': args
    }

    def process_efficient_metrics(efficient):
        """自适应处理嵌套的性能指标（跳过shape字段）"""
        processed = efficient.copy()

        # 需要跳过的字段列表
        skip_keys = {'shape', 'in', 'out', 'flops_fwd_bwd', '2D_flops', 'activation', 'params'  }  # 添加需要跳过的字段

        for metric_key in list(processed.keys()):
            # 跳过形状相关字段
            if metric_key in skip_keys:
                processed.pop(metric_key)
                continue

            # 处理字典类型的指标
            if isinstance(processed[metric_key], dict):
                metric_data = processed.pop(metric_key)

                for dim, values in metric_data.items():
                    new_key = f"{metric_key}_{dim}"

                    # 跳过非数值字段
                    if dim in skip_keys or not isinstance(values, (int, float, list)):
                        processed[new_key] = values
                        continue

                    # 对数值列表求和，标量直接存储
                    if isinstance(values, list):
                        processed[new_key] = sum(values)
                    else:
                        processed[new_key] = values

        return processed

    def process_test_case(test_case):
        """处理测试用例数据，返回扁平化的字典"""
        def flatten_omega_config(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, DictConfig)):
                    items.extend(flatten_omega_config(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        test_case_dict = OmegaConf.to_container(test_case, resolve=True)

        # 处理 input 字段
        if 'input' in test_case_dict:
            input_list = test_case_dict['input']
            shape_params = []
            operation_params = {}

            # 分离形状参数和操作参数
            for item in input_list:
                if isinstance(item, (list, ListConfig)):
                    shape_params.append(list(item))
                elif isinstance(item, (dict, DictConfig)):
                    flat_dict = flatten_omega_config(item)
                    for key, value in flat_dict.items():
                        # 生成唯一键名（自动添加后缀）
                        new_key = key
                        suffix = 1
                        while new_key in operation_params:
                            new_key = f"{key}_{suffix}"
                            suffix += 1
                        operation_params[new_key] = value

            # 生成 input_shape
            if shape_params:
                test_case_dict["input_shape"] = (
                    str(shape_params[0])
                    if len(shape_params) == 1
                    else str(shape_params)
                )

            # 合并操作参数到顶层（不覆盖已有字段）
            for key, value in operation_params.items():
                if key not in test_case_dict:
                    test_case_dict[key] = value

            # 移除原始 input 字段
            del test_case_dict['input']

        return test_case_dict

    process_test_case_dict = process_test_case(test_case)
    precision = process_test_case_dict['precision'] if 'precision' in process_test_case_dict else args.training.precision
    if 'precision' in process_test_case_dict :
        del process_test_case_dict['precision']
    if 'name' in process_test_case_dict:
        del process_test_case_dict['name']
    result['kwargs'] = process_test_case_dict

    result["config"]['dtype'] = precision
    if 'hardware' in result['config']:
        del result['config']['hardware']
    del result['config']['precision']

    if acc:
        for line in acc:
            data.append(deepcopy(result))
            data[-1]["acc"] = {}
            for i, value in enumerate(line):
                data[-1]["acc"][config['format'][i]] = value
    else:
        result["duration"] = {
            'mean': _mean,
            'min': _min,
            'max': _max
        }
        # Process theory latency
        if theory:
            result['theory'] = {
                'latency': theory.get('latency', None)
            }
        if efficient:
            result['efficient'] = process_efficient_metrics(efficient)
        data = [result]

    return data


report = Report()
