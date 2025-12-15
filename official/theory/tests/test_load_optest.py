import pytest
from theory.core.loaders import loader
from theory.core import Op


@pytest.mark.parametrize("path,filters", [("../data/0922_bs48_1100_op_dump_merge-fa-pad_gather-cache", ['flash_attn::varlen_fwd'])])
def test_optrace_loader(
    path: str,
    filters: list,
):
    optest_loader = loader.load('optest', cfg={'path': path, 'filter': filters})
    for op_name in optest_loader:
        print('='*50, op_name, '='*50)
        for i, case in enumerate(optest_loader[op_name]):
            print(i, op_name, case)