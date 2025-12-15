import pytest
from theory.core.loaders import loader
from theory.core.backend import backend


# @pytest.mark.parametrize("model, path", ["DeepSeek_V3", "/data/gpu/nsys/timeline_H20SXMML_DEEPSEEK_V3_no_graph_nsys-shape.nsys-rep"])
@pytest.mark.parametrize("model, path", [("Qwen3_8B", "data/gpu/nsys/timeline_H20TT_train_qwen3_8B-4kbf16_is_causal_nsys-shape.nsys-rep"),])
@pytest.mark.parametrize("detection", [{
        'Dataloader&Emb': {'FWD': [61604, 61648], 'BWD': [67370, 67417]},
        'Attention': {'FWD': [61648, 61683], 'BWD': [63459, 63529]},
        # 'MLP': {'FWD': [61683, 61695], 'BWD': [63425, 63459]},
        'MLP': {'FWD': [61683, 61695], 'BWD': [63529, 63569]},
        'Post': {'FWD': [63340, 63347], 'BWD': [63394, 63425]},
        'Loss': {'FWD': [63347, 63376], 'BWD': [63376, 63394]},
        'Optim': {'Step': [67417, 67452]},
    },])
def test_nsys_loader(
    model: str,
    path: str,
    detection: dict=None,
):
    profile_backend = backend.get('profile', 'nsys')
    nsys_loader = loader.load('nsys', profile_backend, cfg={'path': path, 'modules': detection, 'model': model})