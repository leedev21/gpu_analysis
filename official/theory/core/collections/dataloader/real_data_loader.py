import torch
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import re
import logging
from omegaconf import DictConfig, ListConfig
from .data_loader import BatchSampler, clone_by_loop_process, get_tensor_info_by_loop_process, configs


class RealDataLoader:
    """
    从 dump.json 文件加载真实数据的数据加载器
    """
    
    def __init__(self, dump_json_path: str, pt_data_dir: str, enable_real_data: bool = False):
        """
        初始化真实数据加载器
        
        Args:
            dump_json_path: dump.json 文件路径
            pt_data_dir: pt 文件目录路径
            enable_real_data: 是否启用真实数据加载
        """
        self.dump_json_path = dump_json_path
        self.pt_data_dir = pt_data_dir
        self.enable_real_data = enable_real_data
        self.dump_data = None
        self.logger = logging.getLogger(__name__)
        
        if self.enable_real_data:
            self._load_dump_data()
    
    def _load_dump_data(self):
        """加载 dump.json 数据"""
        try:
            if not os.path.exists(self.dump_json_path):
                self.logger.warning(f"dump.json 文件不存在: {self.dump_json_path}")
                self.enable_real_data = False
                return
                
            with open(self.dump_json_path, 'r', encoding='utf-8') as f:
                self.dump_data = json.load(f)
                
            self.logger.info(f"成功加载 dump.json 文件: {self.dump_json_path}")
            
        except Exception as e:
            self.logger.error(f"加载 dump.json 文件失败: {e}")
            self.enable_real_data = False
    
    def _extract_operator_name(self, full_name: str) -> str:
        """
        从完整的算子名称中提取核心算子名称
        
        Args:
            full_name: 完整的算子名称，如 "VLLM.fused_add_rms_norm_per_token_group_quant_fp8.0.forward"
            
        Returns:
            核心算子名称，如 "fused_add_rms_norm_per_token_group_quant_fp8"
        """
        # 匹配规则：以 . 开始，以 . 结束的部分中提取算子名称
        pattern = r'\.([^.]+)\..*\.forward$'
        match = re.search(pattern, full_name)
        if match:
            return match.group(1)
        
        # 如果没有匹配到，尝试简单的分割
        parts = full_name.split('.')
        if len(parts) >= 2:
            return parts[1]
        
        return full_name
    
    def _find_operator_data(self, op_name: str) -> Optional[Dict]:
        """
        在 dump.json 中查找指定算子的数据
        
        Args:
            op_name: 算子名称
            
        Returns:
            算子数据字典，如果找不到则返回 None
        """
        if not self.dump_data or 'data' not in self.dump_data:
            return None
        
        # 在 dump.json 的 data 字段中查找匹配的算子
        for full_op_name, op_data in self.dump_data['data'].items():
            extracted_name = self._extract_operator_name(full_op_name)
            if extracted_name == op_name:
                return op_data
        
        return None
    
    def _load_pt_file(self, data_name: str) -> Optional[torch.Tensor]:
        """
        加载 pt 文件数据
        
        Args:
            data_name: pt 文件名
            
        Returns:
            加载的张量数据，如果加载失败则返回 None
        """
        try:
            pt_file_path = os.path.join(self.pt_data_dir, data_name)
            if not os.path.exists(pt_file_path):
                self.logger.warning(f"pt 文件不存在: {pt_file_path}")
                return None
            
            tensor = torch.load(pt_file_path, map_location='cpu')
            self.logger.debug(f"成功加载 pt 文件: {pt_file_path}, shape: {tensor.shape}")
            return tensor
            
        except Exception as e:
            self.logger.error(f"加载 pt 文件失败 {data_name}: {e}")
            return None
    
    def _create_real_data_batch(self, op_name: str, test_case: DictConfig, config: DictConfig) -> Optional[Dict]:
        """
        从真实数据创建批次
        
        Args:
            op_name: 算子名称
            test_case: 测试用例配置
            config: 配置信息
            
        Returns:
            包含 args 和 kwargs 的字典，如果创建失败则返回 None
        """
        # 查找算子数据
        op_data = self._find_operator_data(op_name)
        if not op_data:
            self.logger.warning(f"未找到算子数据: {op_name}")
            return None
        
        # 获取输入参数
        input_args = op_data.get('input_args', [])
        if not input_args:
            self.logger.warning(f"算子 {op_name} 没有输入参数")
            return None
        
        # 检查测试用例的输入配置结构
        if not test_case.input or len(test_case.input) == 0:
            self.logger.warning(f"测试用例 {test_case.name} 没有输入配置")
            return None
        
        # 检查第一个输入是否是字典（关键字参数模式）
        first_input = test_case.input[0]
        if isinstance(first_input, (dict, DictConfig)):
            # 这是关键字参数模式
            return self._create_kwargs_batch(input_args, first_input, test_case, config)
        else:
            # 这是位置参数模式
            return self._create_args_batch(input_args, test_case, config)
    
    def _create_kwargs_batch(self, input_args: List, input_config: DictConfig, test_case: DictConfig, config: DictConfig) -> Optional[Dict]:
        """
        创建基于关键字参数的数据批次
        
        Args:
            input_args: dump.json 中的输入参数列表
            input_config: 测试用例的输入配置
            test_case: 测试用例配置
            config: 配置信息
            
        Returns:
            包含 args 和 kwargs 的字典
        """
        real_kwargs = {}
        
        # 从 input_config 中提取参数名称顺序
        param_names = list(input_config.keys())
        
        # 将 input_args 按照参数名称映射到 kwargs
        for i, input_arg in enumerate(input_args):
            if i < len(param_names):
                param_name = param_names[i]
                
                if isinstance(input_arg, dict):
                    # 如果是张量参数
                    if input_arg.get('type') == 'torch.Tensor':
                        data_name = input_arg.get('data_name')
                        if data_name:
                            # 从 pt 文件加载真实数据
                            tensor = self._load_pt_file(data_name)
                            if tensor is not None:
                                real_kwargs[param_name] = tensor
                            else:
                                self.logger.warning(f"无法加载张量数据: {data_name}")
                                return None
                        else:
                            self.logger.warning(f"张量参数 {param_name} 没有 data_name")
                            return None
                    else:
                        # 处理其他类型的参数（如 float, int）
                        param_type = input_arg.get('type', 'unknown')
                        param_value = input_arg.get('value')
                        
                        if param_type == 'float':
                            real_kwargs[param_name] = float(param_value)
                        elif param_type == 'int':
                            real_kwargs[param_name] = int(param_value)
                        elif param_type == 'bool':
                            real_kwargs[param_name] = bool(param_value)
                        else:
                            self.logger.warning(f"不支持的参数类型: {param_type}")
                            return None
                else:
                    # 直接使用参数值
                    real_kwargs[param_name] = input_arg
            else:
                self.logger.warning(f"输入参数数量 ({len(input_args)}) 超过了配置参数数量 ({len(param_names)})")
                break
        
        self.logger.info(f"成功创建真实数据批次 {test_case.name}: {len(real_kwargs)} 个关键字参数")
        return {
            'args': [],
            'kwargs': real_kwargs
        }
    
    def _create_args_batch(self, input_args: List, test_case: DictConfig, config: DictConfig) -> Optional[Dict]:
        """
        创建基于位置参数的数据批次
        
        Args:
            input_args: dump.json 中的输入参数列表
            test_case: 测试用例配置
            config: 配置信息
            
        Returns:
            包含 args 和 kwargs 的字典
        """
        real_args = []
        
        for i, input_arg in enumerate(input_args):
            if isinstance(input_arg, dict):
                # 如果是张量参数
                if input_arg.get('type') == 'torch.Tensor':
                    data_name = input_arg.get('data_name')
                    if data_name:
                        # 从 pt 文件加载真实数据
                        tensor = self._load_pt_file(data_name)
                        if tensor is not None:
                            real_args.append(tensor)
                        else:
                            self.logger.warning(f"无法加载张量数据: {data_name}")
                            return None
                    else:
                        self.logger.warning(f"张量参数 {i} 没有 data_name")
                        return None
                else:
                    # 处理其他类型的参数（如 float, int）
                    param_type = input_arg.get('type', 'unknown')
                    param_value = input_arg.get('value')
                    
                    if param_type == 'float':
                        real_args.append(float(param_value))
                    elif param_type == 'int':
                        real_args.append(int(param_value))
                    elif param_type == 'bool':
                        real_args.append(bool(param_value))
                    else:
                        self.logger.warning(f"不支持的参数类型: {param_type}")
                        return None
            else:
                # 直接使用参数值
                real_args.append(input_arg)
        
        self.logger.info(f"成功创建真实数据批次 {test_case.name}: {len(real_args)} 个位置参数")
        return {
            'args': real_args,
            'kwargs': {}
        }
    
    def get_real_data_batch(self, test_case: DictConfig, config: DictConfig) -> Optional[Dict]:
        """
        获取真实数据批次
        
        Args:
            test_case: 测试用例配置
            config: 配置信息
            
        Returns:
            包含 args 和 kwargs 的字典，如果获取失败则返回 None
        """
        if not self.enable_real_data:
            return None
        
        # 提取算子名称
        op_name = test_case.name
        if '::' in op_name:
            op_name = op_name.split('::')[1]
        
        # 创建真实数据批次
        return self._create_real_data_batch(op_name, test_case, config)


class RealDataBatchSampler(BatchSampler):
    """
    扩展的批次采样器，支持真实数据加载
    """
    
    def __init__(self, test_case, config, seed, real_data_loader: Optional[RealDataLoader] = None):
        """
        初始化批次采样器
        
        Args:
            test_case: 测试用例配置
            config: 配置信息
            seed: 随机种子
            real_data_loader: 真实数据加载器（可选）
        """
        self.real_data_loader = real_data_loader
        self.use_real_data = False
        
        # 获取正确的设备类型（与原始 BatchSampler 一致）
        global configs
        device = 'cpu' if torch.distributed.is_initialized() else configs['DEVICE']
        
        # 首先尝试使用真实数据
        if self.real_data_loader:
            real_batch = self.real_data_loader.get_real_data_batch(test_case, config)
            if real_batch:
                self.use_real_data = True
                self.test_case = test_case
                self.out_mapping = test_case.output
                self.config = config
                self.seed = seed
                self.by_new_style = True
                self.scales = []
                self._out_info = []
                
                # 设置真实数据并移动到正确设备
                self.args = real_batch['args']
                self.kwargs = real_batch['kwargs']
                self._args = real_batch['args']
                self._kwargs = real_batch['kwargs']
                self.len_kwargs = len(self._kwargs)
                
                # 将数据移动到正确设备
                self._move_to_device(device)
                
                # 如果有输出映射，执行 by_list 处理
                if self.test_case.output:
                    self.by_list()
                
                # 清理缓存并打印调试信息
                self.scales.clear()
                self.print_randn()
                
                # 记录使用真实数据
                logging.getLogger(__name__).info(f"使用真实数据进行测试: {test_case.name}")
                return
        
        # 如果真实数据加载失败，使用原始的随机数据生成
        super().__init__(test_case, config, seed)
    
    def _move_to_device(self, device):
        """将数据移动到指定设备"""
        from moprobe.utils import to_device
        
        # 移动位置参数
        if self._args:
            self._args = to_device(torch.device(device), self._args)
            self.args = self._args  # 直接引用，保持一致性
        
        # 移动关键字参数
        if self._kwargs:
            self._kwargs = to_device(torch.device(device), self._kwargs)
            self.kwargs = self._kwargs  # 直接引用，保持一致性
    
    def by_list(self):
        """将 kwargs 转换为 args 列表形式（与原始 BatchSampler 一致）"""
        if self.use_real_data:
            for k, v in self.kwargs.items():
                if k in self.out_mapping:
                    self.out_mapping[self.out_mapping.index(k)] = len(self.args)
                self.args.append(v)
            self.len_kwargs = 0
            # 同步 _args 和 _kwargs
            self._args = self.args
            self._kwargs = {}
        else:
            super().by_list()
    
    def to(self, device, use_cpu_hight_precision=False, fp8_transfer=False):
        """将数据移动到指定设备"""
        if self.use_real_data:
            from moprobe.utils import to_device
            self._args = to_device(torch.device(device), self._args, use_cpu_hight_precision, fp8_transfer)
            if self.has_kwargs():
                self._kwargs = to_device(torch.device(device), self._kwargs, use_cpu_hight_precision, fp8_transfer)
            
            # 同步 args 和 kwargs，保持引用一致性
            self.args = self._args
            self.kwargs = self._kwargs
            return self
        else:
            return super().to(device, use_cpu_hight_precision, fp8_transfer)
    
    def has_kwargs(self):
        """检查是否有关键字参数"""
        if self.use_real_data:
            return self.len_kwargs > 0
        else:
            return super().has_kwargs()
    
    def get_input(self):
        """获取输入数据"""
        if self.use_real_data:
            return self._args, self._kwargs
        else:
            return super().get_input()
    
    def print_randn(self):
        """打印批次数据信息"""
        if self.use_real_data:
            print('\n---------------------- create batch (real data) -----------------------')
            for i, v in enumerate(self.args):
                if isinstance(v, torch.Tensor):
                    print(i, v.device, v.shape, v.dtype)
                else:
                    print(i, v)
            if self.has_kwargs():
                for k, v in self.kwargs.items():
                    if isinstance(v, torch.Tensor):
                        print(k, v.device, v.shape, v.dtype)
                    else:
                        print(k, v)
        else:
            super().print_randn()
    
    def clone(self):
        """克隆批次采样器"""
        if self.use_real_data:
            # 对于真实数据，与原始 BatchSampler 保持一致：克隆数据但返回自身
            self._args = clone_by_loop_process(self.args)
            self._kwargs = clone_by_loop_process(self.kwargs)
            self._out_info = []
            return self
        else:
            return super().clone()
    
    def get_output(self):
        """获取输出数据，支持关键字参数模式"""
        if self.use_real_data:
            # 对于真实数据，需要根据输出映射来获取输出
            if self.out_mapping:
                if isinstance(self.out_mapping[0], int):
                    # 位置参数模式
                    result_list = []
                    for index in self.out_mapping:
                        item = self._args[index]
                        if isinstance(item, list):
                            # Handle tensor list - move each tensor to CPU
                            result_list.append([tensor.detach().to('cpu') for tensor in item])
                            self._out_info.append(self.get_tensor_info(item))
                        else:
                            # Handle single tensor
                            self._out_info.append(self.get_tensor_info(item))
                            result_list.append(item.detach().to('cpu'))
                    return result_list
                else:
                    # 关键字参数模式
                    self._out_info.extend([self.get_tensor_info(item, k) for k, item in self._kwargs.items() if k in self.out_mapping])
                    return [self._kwargs[key].detach().to('cpu') for key in self.out_mapping]
            return None
        else:
            # 使用父类的实现
            return super().get_output()
    
    def check_output(self, output):
        """检查输出数据并填充 _out_info"""
        if self.use_real_data:
            # 对于真实数据，使用与原始 BatchSampler 相同的逻辑
            if isinstance(output, list):
                for item in output:
                    self._out_info.append(self.get_tensor_info(item))
            elif isinstance(output, dict):
                for k, item in output.items():
                    self._out_info.append(self.get_tensor_info(item, k))
            else:
                self._out_info.append(self.get_tensor_info(output))
        else:
            # 使用父类的实现
            super().check_output(output)
    
    def get_case_info(self):
        """获取测试用例信息"""
        if self.use_real_data:
            # 对于真实数据，返回与原始 BatchSampler 相同格式的信息
            # 不在这里创建 _out_info，让它通过正常的 check_output 或 get_output 流程来填充
            return {
                'name': self.test_case.name,
                'seed': self.seed,
                'test_case': self.test_case,
                'config': self.config,
                'out': self._out_info
            }
        else:
            return super().get_case_info() 