import torch
import importlib
from typing import List, Tuple

# Pre-import packages to avoid repeated imports in get_op function
try:
    import apex
except ImportError:
    apex = None

try:
    import vllm
except ImportError:
    print('vllm not found')
    vllm = None



try:
    import torch_custom_op_native
except ImportError:
    print('custom_op_native not found')
    torch_custom_op_native = None

def get_op(package, op_name):
    # No need for imports here anymore as they are handled at module level
    if 'ops' in package:
        lib = getattr(torch.ops, package.split('.')[-1])
        return getattr(lib, op_name)
    package_obj = importlib.import_module(package)
    items = op_name.split('.')
    try:
        module = importlib.import_module(f"{'.'.join(items[:-1])}")
    except:
        items.insert(0, package)
        module = importlib.import_module(f"{'.'.join(items[:-1])}")
    return getattr(module, items[-1])
