import sys
import subprocess
from pathlib import Path


def get_apex_path():
    """Find Transformer Engine install path using pip"""
    command = [sys.executable, "-m", "pip", "show", "apex"]
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    result = result.stdout.replace("\n", ":").split(":")
    if result and len(result) > 1:
        key = 'Editable project location' if 'Editable project location' in result else 'Location'
        return result[result.index(key) + 1].strip()
    else:
        return ""

apex_so = [
    'amp_C', 'apex_C', 'distributed_adam_cuda', 'fast_layer_norm', 'fused_adam_cuda', 'fused_dense_cuda',
    'fused_layer_norm_cuda', 'fused_rotary_positional_embedding', 'fused_weight_gradient_mlp_cuda',
    'generic_scaled_masked_softmax_cuda', 'group_norm_cuda', 'mlp_cuda', 'scaled_masked_softmax_cuda',
    'scaled_softmax_cuda', 'scaled_upper_triang_masked_softmax_cuda', 'syncbn'
]

def _load_apex_torch():
    so_path = get_apex_path()
    if 'dist-packages' not in so_path:
        return f'path:{so_path}'
    return apex_so
