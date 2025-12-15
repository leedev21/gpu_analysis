import pytest
from theory.core import Op


@pytest.mark.parametrize("op_name", ['op::EmbeddingFwd', 'op::EmbeddingBwd'])
def test_basic_op(
    op_name: str,
):
    optrace_obj = Op(op_name, [128, 256], backend='basic')