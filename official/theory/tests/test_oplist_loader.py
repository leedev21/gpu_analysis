import pytest
from theory.core.loaders import loader
from theory.core import Op


@pytest.mark.parametrize("path", ["../../training/trace/model/deepdeek_v3_training_te2_5_blockwise_fp8.trace"])
def test_oplist_loader(
    path: str,
):
    report_map = {
        'op::RMSNormFwd': {'name': 'layernorm', 'shape': ['N', 'H']},
        'op::MM': {'name': 'gemm.shared', 'shape': ['B', 'M', 'K', 'N']},
        'op::AttnFwd': {'name': 'sdpa', 'shape': ['N', 'B', 'K', 'Hqk']},
        'op::SigmoidFwd': {'name': 'sigmoid', 'shape': ['N', 'H']},
        'op::Add': {'name': 'add', 'shape': ['N', 'H']},
        'op::Silu': {'name': 'silu', 'shape': ['N', 'H']},
        'op::Mul': {'name': 'mul', 'shape': ['N', 'H']},
        # 'op::Silu': {'name': 'silu', ['N', 'H']},
        # 'op::AttnBwd': {'name': 'layernorm', ['N', 'H']},
        # 'op::RMSNormBwd': {'name': 'layernorm', ['N', 'H']},
        # 'op::Topk': {'name': 'layernorm', ['N', 'H']},
    }
    dtype_map = {
        'BFloat16': 'BF16',
        'Float': 'FP32',
        'Int': 'Int',
    }
    debug_cfg =  {'te::generic_gemm': ['parse', 'trace']}
    optrace_loader = loader.load('oplist', cfg={'path': path, 'debug': debug_cfg})
    report = []
    print('='*50, 'final report', '='*50)
    for op_name in optrace_loader:
        print('='*50, op_name, '='*50)
        for traced_op in optrace_loader[op_name]:
            optrace_obj = Op(traced_op, backend='trace', debug=debug_cfg)
            if optrace_obj[0] and optrace_obj[0] in report_map:
                # print('\t', optrace_obj)
                shape = [optrace_obj[2].get(k, 1) for k in report_map[optrace_obj[0]]['shape']]
                case = {"optype": report_map[optrace_obj[0]]['name'],
                               "shape": shape,
                               "dtype": dtype_map[optrace_obj[3]],
                               'model_name': "deepseekv3-4K",
                               'tag': ["train"]}
                if case not in report:
                    report.append(case)
            if optrace_obj[0] and any(k in op_name for k in ['mm', 'rmsnorm_fwd', 'rmsnorm_bwd', 'topk', 'fused_attn_bwd', 'fused_attn_fwd', 'embedding']):
                print(optrace_obj)
    for case in report:
        print(case)