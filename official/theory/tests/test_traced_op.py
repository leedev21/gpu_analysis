import pytest
from theory.core import Op
from theory.core.common.op import LaunchOp
from theory.core.utils import Parser


@pytest.mark.parametrize("case", ['aten::embedding(weight:<129280x7168xbf16>{7168, 1}+1741560832, indices:<1x4096xi64>{4096, 1}) -> <1x4096x7168xbf16>{29360128, 7168, 1}',
                                  'aten::embedding_dense_backward(grad_output:<1x4096x7168xbf16>{7168, 7168, 1}, indices:<1x4096xi64>{4096, 1}, num_weights:129280:int, padding_idx:-1:int, scale_grad_by_freq:False:bool) -> <129280x7168xbf16>{7168, 1}'])
def test_trace_op(
    case: str,
):
    traced_op = LaunchOp(case, Parser)
    optrace_obj = Op(traced_op, backend='trace')
    print(optrace_obj)