import os
import sys
import subprocess
import ctypes
import glob
import sysconfig
import importlib
from pathlib import Path
import platform


def get_package_path(package):
    """Find Transformer Engine install path using pip"""
    command = [sys.executable, "-m", "pip", "show", package]
    result = subprocess.run(command, capture_output=True, check=True, text=True)
    result = result.stdout.replace("\n", ":").split(":")
    if result and len(result) > 1:
        return Path(result[result.index("Location") + 1].strip())
    else:
        return None


def _load_jit_library(package, swap_funcs):
    if isinstance(package, Path):
        so_path = str(package)
    else:
        so_path = get_package_path(package) / package
    func_list = []
    for swap_func in swap_funcs:
        command = ["grep", "-A", "1", swap_func, str(so_path), "-r"]
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        for line in result.stdout.split('\n'):
            if 'def' in line:
                path = line.split('.py')[0]
                father = path[path.index(package):].replace('/', '.')
                line = line.split('def ')[1].split('(')[0]
                func_list.append(father + '.' + line.strip())
                # yield module, item, getattr(module, item)
    return func_list


if __name__ == "__main__":
    # res = _load_jit_library('vllm', 'triton.jit')
    res = _load_jit_library(Path('/usr/local/lib/python3.12/dist-packages/transformer_engine'), ['jit_fuser', 'dropout_fuser'])
    for line in res:
        print(line)


import os
import sys
import subprocess
import ctypes
import glob
import sysconfig
import importlib
from pathlib import Path
import platform
from importlib import import_module

def get_package_path(package):
    """Find Transformer Engine install path using pip"""
    command = [sys.executable, "-m", "pip", "show", package]
    result = subprocess.run(command, capture_output=True, check=True, text=True)
    result = result.stdout.replace("\n", ":").split(":")
    if result and len(result) > 1:
        return Path(result[result.index("Location") + 1].strip())
    else:
        return None


def _load_jit_library(package, swap_funcs):
    if isinstance(package, Path):
        so_path = str(package)
        package = so_path.split('/')[-1]
    else:
        so_path = get_package_path(package) / package
    func_list = []
    for swap_func in swap_funcs:
        command = ["grep", "-A", "1", swap_func, str(so_path), "-r"]
        result = subprocess.run(command, capture_output=True, check=True, text=True)
        for line in result.stdout.split('\n'):
            if 'def' in line:
                path = line.split('.py')[0]
                father = path[path.index(package):].replace('/', '.')
                op_name = line.split('def ')[1].split('(')[0].strip()
                module = import_module(father)
                yield module, op_name, getattr(module, op_name)


if __name__ == "__main__":
    # res = _load_jit_library('vllm', 'triton.jit')
    res = _load_jit_library(Path('/usr/local/lib/python3.12/dist-packages/transformer_engine'), ['jit_fuser', 'dropout_fuser'])
    for module, op_name, op in res:
        op_name = op._fn_name if hasattr(op, '_fn_name') else op.__name__
        print('hook --> triton::', op_name, '<==>', op, op.__class__.__name__)