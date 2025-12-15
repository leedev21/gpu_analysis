#!/usr/bin/env python3
"""
使用真实数据进行算子测试的示例脚本

用法:
    python run_with_real_data.py \
        --dump_json_path /path/to/dump.json \
        --pt_data_dir /path/to/pt/files \
        --op_config vllm/all.yaml \
        --case_file launcher/tools/all.csv

或者使用默认路径:
    python run_with_real_data.py \
        --dump_json_path data/1/device-1card-dispatch-635.6/step1/rank0/dump.json \
        --pt_data_dir data/1/device-1card-dispatch-635.6/step1/rank0/dump_tensor_data \
        --op_config vllm/all.yaml
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def main():
    parser = argparse.ArgumentParser(description='使用真实数据进行算子测试')
    parser.add_argument('--dump_json_path', required=True, help='dump.json 文件路径')
    parser.add_argument('--pt_data_dir', required=True, help='pt 文件目录路径')
    parser.add_argument('--op_config', default='vllm/all.yaml', help='算子配置文件')
    parser.add_argument('--case_file', default=None, help='测试用例文件')
    parser.add_argument('--run_config', default='real_data_check', help='运行配置')
    parser.add_argument('--draw', action='store_true', help='是否绘制图表')
    parser.add_argument('--save_output', action='store_true', help='是否保存输出到 pt 文件')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.dump_json_path):
        print(f"错误: dump.json 文件不存在: {args.dump_json_path}")
        sys.exit(1)
    
    if not os.path.exists(args.pt_data_dir):
        print(f"错误: pt 文件目录不存在: {args.pt_data_dir}")
        sys.exit(1)
    
    # 构建命令行参数
    cmd_args = [
        'python', 'scheduler.py',
        f'op={args.op_config}',
        f'run={args.run_config}',
        f'run.enable_real_data=True',
        f'run.dump_json_path={args.dump_json_path}',
        f'run.pt_data_dir={args.pt_data_dir}',
    ]
    
    if args.case_file:
        cmd_args.append(f'load_case_file={args.case_file}')
    
    if args.draw:
        cmd_args.append('run.draw=True')
    
    if args.save_output:
        cmd_args.append('run.save_output_by_pt=True')
    
    # 显示要执行的命令
    print("执行命令:")
    print(" ".join(cmd_args))
    print()
    
    # 切换到 launcher 目录
    os.chdir(current_dir)
    
    # 执行命令
    import subprocess
    result = subprocess.run(cmd_args, capture_output=False)
    
    return result.returncode

if __name__ == '__main__':
    sys.exit(main()) 