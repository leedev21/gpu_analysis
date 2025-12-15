import pytest
from theory.core.loaders import loader
from theory.core import Op


# @pytest.mark.parametrize("path", ["../../training/trace/model/deepseek_v3_training.graph"])
@pytest.mark.parametrize("path", ["../data/torchtitan_qwen3_8B/pt0/1_1.trace"])
def test_graphtrace_loader(
    path: str,
):
    graphtrace_loader = loader.load('graphtrace', cfg={'path': path,})