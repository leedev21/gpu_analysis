import csv
import json
from collections import defaultdict
import os

class Utils:
    @staticmethod
    def read_file(base_path, filename):
        file_path = os.path.join(base_path, filename)
        with open(file_path, 'r') as f:
            return f.readlines()

class CSVToJSONConverter:
    def __init__(self, base_config_path):
        self.base_config_path = base_config_path
        self.utils = Utils()
        self.torch_suport_op = [k.strip().lower() for k in self.utils.read_file(self.base_config_path, 'support_by_torch.txt')]

    def clean_range(self, range_str):
        return range_str.strip().strip('#').strip()

    def format_duration(self, duration_ns):
        if duration_ns >= 1e6:  # 大于等于1ms
            return f"{duration_ns/1e6:.3f}ms"
        elif duration_ns >= 1e3:  # 大于等于1μs
            return f"{duration_ns/1e3:.3f}us"
        else:  # 小于1μs
            return f"{duration_ns:.3f}ns"

    def csv_to_json(self, csv_file, json_file):
        # 读取CSV文件
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)

        # 创建一个默认字典来存储合并后的数据
        merged_data = defaultdict(lambda: defaultdict(list))

        # 处理每一行数据
        for row in data:
            op_name = row['Range'].strip()
            # print("op_name",op_name)
            is_support_by_torch = op_name in self.torch_suport_op


            op_type = None if ':' not in op_name else op_name.split(':')[0]
            if op_name.startswith('nvte'):
                op_type = 'TE'
            elif op_name.startswith('FlashAttn'):
                op_type = 'FlashAttn'
            elif op_type is None:
                op_type = 'customer'

            if '::' in op_name:
                _, op_name = op_name.split('::', 1)

            # 清理op_name
            op_name = self.clean_range(op_name)

            if not op_name:  # 如果清理后的op_name为空，跳过这一行
                continue

            # print("self.torch_suport_op",self.torch_suport_op)
            # 创建op_info字典
            op_info = {
                "size": row['Size'].strip().replace('sizes = ', ''),
                "gpu_duration": self.format_duration(float(row['Avg (ns)'].strip())),
                "has_kernel": None,  # 假设所有操作都有kernel
                "has_aten": None,  # 假设aten为0
                "support_by_torch": is_support_by_torch,  # 根据op_name是否在torch支持的操作列表中来判断
                "stride": None,
                "cpu_duration": None,
                "cuda_theory": None
            }

            # if op_info["support_by_torch"]:
            #     print(f"Operation {op_name} is supported by torch.")



            # 将op_info添加到对应的op_type和op_name下
            merged_data[op_type][op_name].append(op_info)

        # 将合并后的数据转换为最终的JSON格式
        final_data = []
        for op_type, ops in merged_data.items():
            for op_name, op_infos in ops.items():
                final_data.append({
                    "op_type": op_type,
                    "op_name": op_name,
                    "op_info": op_infos
                })

        # 写入JSON文件
        with open(json_file, 'w') as f:
            json.dump(final_data, f, indent=4)

# 使用函数
base_config_path = './'  # 请替换为实际的配置文件目录路径
import sys
import os

import os

directory = sys.argv[1]

for filename in os.listdir(directory):
    print("processing file:",filename)
    if filename.endswith('.csv'):
        csv_file = os.path.join(directory, filename)
        json_file = os.path.splitext(filename)[0] + '.json'
        converter = CSVToJSONConverter(base_config_path)
        converter.csv_to_json(csv_file, json_file)
