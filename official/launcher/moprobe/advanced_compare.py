import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from omegaconf.dictconfig import DictConfig
import json
from launcher.utils import get_device


def draw_ulp(title, n, data, path):
    # 绘制结果
    print('\t===============', title, '===============')
    dim = data.shape[-1]
    data = np.reshape(data, (-1, dim))
    x = range(dim)
    try:
        plt.figure()
        for i in range(dim):
            mask = data[:][i] >= 1
            y = data[:][i][mask]
            x = [i for k in range(len(y))]
            s = y / np.max(y) * 100
            plt.scatter(x, y, s)
        # for i in range(data.shape[0]):
        #     plt.scatter(x, data[i])
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('ulp_err')
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(f'{path}/savefig_{title}_{dim}_{n}.png', transparent=False)
    except:
        torch.save(data, f'{path}/savefig_{title}_{dim}_{n}.pt')

def draw_ulp_log(title, n, data, path):
    # 绘制结果
    print('\t===============', title, '===============')
    x = range(280)
    _min, _max, _bins = -151, 127, 280
    torch.save(data, f'{path}/savefig_{title}_{n}.pt')
    for case in data:
        try:
            # plt.xscale('log', base=2)
            # plt.yscale('log', base=2)
            # Calculate scaling factor
            positive_data = data[case]['positive']
            _max = 100 / np.max(positive_data) + 5

            # Create coordinate grids
            i, j = np.meshgrid(np.arange(_bins), np.arange(_bins), indexing='ij')

            # Filter non-zero points and calculate sizes
            mask = positive_data != 0
            x_coords = (i[mask] + _min).ravel()
            y_coords = (j[mask] + _min).ravel()
            sizes = (positive_data[mask] * _max).ravel()

            # Create scatter plot
            plt.figure()
            plt.scatter(x_coords, y_coords, s=sizes, alpha=0.6)

            # Add labels and annotations
            plt.xlabel('target(log2 coordinate)')
            plt.ylabel('ulp(log2 coordinate)')
            plt.annotate('point size: n_ulp / max(n_ulp) * 100%', (0.5, 0.02), xycoords='axes fraction', ha='center', va='center')
            plt.title(f'{title}_{case}')
            plt.grid(True)
            plt.tight_layout()

            if not os.path.exists(path):
                os.makedirs(path)
            plt.savefig(f'{path}/savefig_{title}_{case}_{n}.png', transparent=False)
        except:
            pass

def calculate_tensor_pos_neg(x_tensor, y_tensor):
    _min, _max, _bins = -151, 127, 280

    positive = np.zeros((_bins, _bins))
    negative = np.zeros((_bins, _bins))

    # 将 PyTorch 张量转换为 NumPy 数组
    x_tensor_np = x_tensor.astype(np.float32)
    y_tensor_np = y_tensor.astype(np.float32)

    # 计算对数
    data_positive = np.log2(x_tensor_np).flatten() - _min
    mask_negative = (x_tensor_np < 0)
    data_negative = np.where(mask_negative, 0, np.log2(-x_tensor_np)).flatten() - _min
    y = np.log2(y_tensor_np).flatten() - _min

    for i in range(len(y)):
        if not np.isnan(data_positive[i]) and not np.isneginf(data_positive[i]):
            if not np.isnan(y[i]) and not np.isneginf(y[i]):
                # print("positive: ({}, {})".format(y[i], data_positive[i]))
                positive[int(y[i])][int(data_positive[i])] += 1
            if not np.isnan(data_negative[i]):
                raise ValueError("Both data_positive and data_negative are not NaN at position ({})".format(i))
        else:
            if np.isnan(data_negative[i]) or np.isneginf(data_negative[i]):
                raise ValueError("Both data_positive and data_negative are NaN at position ({})".format(i))
            elif not np.isnan(y[i]) and not np.isneginf(y[i]):
                # print("negative: ({}, {})".format(y[i], data_negative[i]))
                negative[int(y[i])][int(data_negative[i])] += 1

    return positive, negative


def calculate_tensor(x_tensor, y_tensor, fast_mode=True):
    _min, _max, _bins = -151, 127, 280
    _cal = np.zeros((_bins, _bins), dtype=int) # 确保_cal的数据类型是整数

    # 计算对数并展平
    y = np.minimum(np.log2(np.abs(y_tensor)), _max).flatten() - _min
    x = x_tensor.flatten() - _min

    # 筛选有效数据
    mask_x = ~np.isnan(x) & ~np.isneginf(x)
    mask_y = ~np.isnan(y) & ~np.isneginf(y)

    if fast_mode:
        # 综合两个掩码
        combined_mask = mask_x & mask_y

        # 获取有效数据的整数索引
        # 注意：这里需要确保x和y是整数，如果不是，需要进行类型转换
        valid_y_indices = y[combined_mask].astype(int)
        valid_x_indices = x[combined_mask].astype(int)

        # 确保索引在有效范围内
        # 这一步非常重要，防止索引越界
        valid_y_indices = np.clip(valid_y_indices, 0, _bins - 1)
        valid_x_indices = np.clip(valid_x_indices, 0, _bins - 1)

        if len(valid_x_indices) > 0 and len(valid_y_indices) > 0:
            hist, _, _ = np.histogram2d(valid_x_indices, valid_y_indices, bins=[_bins, _bins], range=[[0, _bins], [0, _bins]])
            _cal = hist.astype(int) # histogram2d返回浮点数，需要转回整数
    else:
        for i in range(len(y)):
            if mask_x[i] and mask_y[i]:
                # print("({}, {})".format(y_tensor[i], x[i]))
                _cal[int(x[i])][int(y[i])] += 1
    return _cal


class DrawManager():
    def __init__(self, path):
        self.path = path
        self.title = ''
        self.tensor_id = ''
        self.hw_name = ''

        # Initialize environment information
        env_info = {}

        # Get version information for -sdk, aten, and cc
        try:
            import subprocess
            sdk_version = subprocess.check_output("dpkg -l | grep -sdk | awk '{print $3}'", shell=True).decode().strip()
            aten_version = subprocess.check_output("dpkg -l | grep aten | awk '{print $3}'", shell=True).decode().strip()
            cc_version = subprocess.check_output("dpkg -l | grep cc | awk '{print $3}'", shell=True).decode().strip()
            vllm_version = subprocess.check_output("pip list | grep vllm_device | awk '{print $2}'", shell=True).decode().strip()

            try:
                gpu_info = subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True).decode().strip()
                self.hw_name = gpu_info.split("\n")[0] if gpu_info else "UnknownGPU"
            except:
                self.hw_name = "UnknownGPU"

            env_info['-sdk'] = sdk_version
            env_info['aten'] = aten_version
            env_info['cc'] = cc_version
            env_info['vllm_version'] = vllm_version
            env_info['GPU:'] = self.hw_name
        except Exception as e:
            print("Error getting version information: %s" % e)
            # Set default values if error occurs
            env_info['-sdk'] = 'unknown'
            env_info['aten'] = 'unknown'
            env_info['cc'] = 'unknown'
            env_info['vllm_version'] = 'unknown'
            self.hw_name = 'unknown'

        self.summary = {
            'env': env_info,
            'op_data': {}
        }

        self.data = {}
        self.state = {}
        self.test_case = {}
        self.output_keys = []
        self.fp8_index = {-1: {}}
        self.n = -1
        self.tensor_n = -1
        self.draw_ulp = False
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists('data/pt/op/'):
            os.makedirs('data/pt/op/', exist_ok=True)
        self.summary_file = f'data/pt/op/op_{get_device()}_summary.json'
        self.out_dtype = ''
        self.ulp_dtype = ''
        self.get_dtype = None

    def add_tensor(self, cls, dtype, data):
        if cls == 'ulp_err':
            res = calculate_tensor(*data)
            if self.tensor_id not in self.data or not self.data[self.tensor_id]:
                self.data[self.tensor_id] = {'positive': res, 'dtype': dtype}
            else:
                assert dtype == self.data[self.tensor_id]['dtype']
                self.data[self.tensor_id]['positive'] += res
        else:
            print('not support!', cls)

    def draw(self, args):
        self.n += 1
        self.title = f"{self.state['name']}_{args['hw']}"
        if self.state['rank'] != -1:
            self.title += f"_rank{self.state['rank']}"
        draw_ulp_log(self.title, self.n, self.data, self.path)
        self.data = {}

    def get(self, cls):
        if cls == 'ulp_err':
            return self.data
        elif cls == 'dtype':
            return self.out_dtype
        else:
            print('not support!', cls)

    def clear(self):
        self.data = {}

    def set(self, attr, value):
        setattr(self, attr, value)
        if attr == 'test_case':
            self.fp8_index = {-1: {}}
            if 'output' in self.test_case:
                if isinstance(self.test_case['output'][0], int):
                    self.output_keys = []
                    for i in self.test_case['output']:
                        k = i - len(self.test_case['input']) + 1
                        self.output_keys.append(list(self.test_case['input'][-1].keys())[k] if k >= 0 else i)
                elif isinstance(self.test_case['output'][0], str):
                    self.output_keys = self.test_case['output']
                for i, items in enumerate(self.test_case['input']):
                    if isinstance(items, (dict, DictConfig)) and '_tensor' in items and 'precision' in items and items['precision'] in ['e4m3', 'e5m2']:
                        self.fp8_index[i] = items['precision']
                    elif isinstance(items, (dict, DictConfig)):
                        j = 0
                        for k, v in items.items():
                            if isinstance(v, (dict, DictConfig)) and '_tensor' in v and 'precision' in v and v['precision'] in ['e4m3', 'e5m2']:
                                # self.fp8_index[-1][k] = v['precision']
                                self.fp8_index[j] = v['precision']
                            j += 1
            # print('self.output_keys:', self.output_keys)
            # print('self.fp8_index:', self.fp8_index)

    def set_title(self, name, rank, target_device, i_key, target_dtype, out_dtype):
        self.ulp_dtype = ''
        self.state = {'name': name,
                      'rank': rank,
                      'target_device': target_device,
                      'key': i_key,
                      'out_dtype': out_dtype,
                      'target_dtype': target_dtype,
                      }
        if i_key in self.output_keys:
            self.state['key'] = self.output_keys[i_key]
            try:
                if 'ulp_dtype' in self.test_case['input'][self.state['key']]:
                    self.ulp_dtype = self.get_dtype(self.test_case['input'][self.state['key']]['ulp_dtype'])
            except:
                pass
        self.title = f"{self.state['name']}_{self.hw_name}"
        if self.state['rank'] != -1:
            self.title += f"_rank{self.state['rank']}"
        self.tensor_id = f"{out_dtype}_{target_device}_{target_dtype}_{self.state['key']}"
        self.data[self.tensor_id] = {}
        self.out_dtype = out_dtype

    def save_pt(self, data, case_info):
        self.tensor_n += 1
        if self.draw_ulp:
            case_info['d'] = data
            file_path = 'data/pt/op'
            # Get basic shape representation
            shape = 'x'.join([str(k) for k in case_info['out'][0].get('shape')])
            # Try to get MxKxN format shape
            try:
                M, N = case_info['out'][0].get('shape')
                K = case_info['test_case']['input'][0]['lhs']['_tensor'][1]  # Try to get K value from lhs
                # If K is successfully obtained, use MxKxN format
                shape = "{}x{}x{}".format(M, K, N)  # MxKxN format
            except Exception as e:
                # Keep using the basic shape format if K cannot be extracted
                print(f"Failed to extract shape: {str(e)}")
                pass

            device = case_info['out'][0].get('device')
            name = case_info['name'].split('::')[-1] if '::' in case_info['name'] else case_info['name']
            case = f"{shape}_{case_info['test_case'].precision}"
            file_name = f'{name}_{case}_{self.hw_name}_{device}_{self.tensor_n}.pt'
            torch.save(case_info, os.path.join(file_path, file_name))
            if name not in self.summary['op_data']:
                self.summary['op_data'][name] = {}
            if case not in self.summary['op_data'][name]:
                self.summary['op_data'][name][case] = {}
            if device not in self.summary['op_data'][name][case]:
                self.summary['op_data'][name][case][device] = []
            self.summary['op_data'][name][case][device].append(file_name)

    def save_summary(self):
        if self.draw_ulp:
            with open(self.summary_file, "w") as f:
                json.dump(self.summary, f)

draw_manager = DrawManager('advanced_compare')