from theory.core.common.op import Op as OpBase
from .base import register_op


@register_op('flash_attn')
class Op(OpBase):
    def __init__(self, name, args=None, bwd=False):
        super().__init__(name)
        self.r = args
        self.bwd = bwd
        # input: [[B, np, Q, D], [-1, 32768, K, D], [-1, 32768, V, D]]     Q, K=V
        # output: [32768, 16, 192]                                          KV
        # exec:
        # aten::baddbmm [[B * np, Q, K], [B * np, Q, D], [B * np, D, K]]    gqa: Q != K; decoding: Q != K
        # aten::_softmax [[B, np, Q, K]]                                    decoding Q=1 & K=s_kv_cache
        # aten::bmm [[B * np, Q, K], [B * np, V, D]]
        # input0: B (Batch size)
        # input1: Q (input length of query)
        # input2: D (headdim; headdim_v)
        # input3: KV -1 if KV == Q (input length of Key, Value)
        # input4: Num of Head for Q (nheads_q)
        # input5: Num of Head for KV (nheads_kv)
        # input6: Head Dims for QK (headdim_qk)
        # eg:
        # aten::repeat_interleave [[128, 1, 2, 64], [], [], []] -> 6.656us
        # aten::repeat_interleave [[128, 1, 2, 64], [], [], []] -> 6.592us
        # aten::baddbmm [[14, 128, 128], [14, 128, 64], [14, 64, 128], [], []] -> 8.065us
        # aten::masked_fill_ [[1, 14, 128, 128], [1, 1, 128, 128], []] -> 8.512us
        # aten::_softmax [[1, 14, 128, 128], [], []] -> 9.216us
        # aten::bmm [[14, 128, 128], [14, 128, 64]] -> 9.248us
        # aten::copy_ [[128, 1, 14, 64], []] -> 7.616us
        # mha: nheads_q=nheads_kv; 
        # gqa: nheads_kv=(max(nheads_q // 8, 1);
        # mqa:
        # mla: nheads_kv=1; headdim_v!=headdim; has_qv if rope;
        # gla: headdim_v!=headdim; has_qv if rope;

    def get_mkn(self, shape, decoding, sdpa=False):
        res = {'in': [], 'out': []}
        if isinstance(shape, list):
            if isinstance(shape[0], list):
                if len(shape) == 1:
                    res['b'] = shape[0][0]
                    res['d_v'] = shape[0][-1]
                    if len(shape[0]) == 5:
                        res['s_q'] = shape[0][1]
                        res['s_kv'] = res['s_q']
                        res['d_qk'] = res['d_v']
                        res['nh_q'] = shape[0][-2]
                        res['nh_kv'] = shape[0][-2]
                    else:
                        res['s_q'] = shape[0][-2]
                        res['s_kv'] = res['s_q']
                        res['d_qk'] = res['d_v']
                else:
                    res['b'] = 1
                    res['s_q'] = 1
                    res['s_kv'] = 1
                    res['d_v'] = shape[0][-1]
                    res['d_qk'] = shape[1][-1]
                    if sdpa:
                        for i in range(2, 4):
                            if shape[0][-i] == shape[1][-i] and res['b'] == 1:
                                res['b'] *= shape[0][-i]
                            else:
                                res['s_q'] *= shape[0][-i]
                                res['s_kv'] *= shape[1][-i]
                    else:
                        res['nh_q'] = shape[0][-2]
                        res['nh_kv'] = shape[1][-2]
                        res['s_q'] = shape[0][-3]
                        res['s_kv'] = shape[1][-3]
                    if len(shape[0]) == 4:
                        res['b'] *= shape[0][0]
                res['in'] = shape
                res['out'] = [res['b'] * res['nh_q'], res['s_q'], res['d_v']]
            else:
                res['b'] = shape[0]
                res['s_q'] = shape[1]
                res['d_v'] = shape[2]
                res['s_kv'] = shape[1]
                res['nh_q'] = 1
                res['nh_kv'] = 1
                res['d_qk'] = shape[2]
                if len(shape) >= 4 and shape[3] != -1:
                    if decoding:
                        res['s_kv'] += shape[3]
                    else:
                        res['s_kv'] = shape[3]
                if len(shape) >= 5 and shape[4] != -1:
                    res['nh_q'] = shape[4]
                    if len(shape) >= 6 and shape[5] != -1:
                        res['nh_kv'] = shape[5]
                    else:
                        res['nh_kv'] = res['nh_q']
                if len(shape) == 7 and shape[6] != -1:
                    res['d_qk'] = shape[6]
                res['in'] =  [[res['b'], res['s_q'], res['d_qk']],
                              [res['b'], res['s_kv'], res['d_qk']],
                              [res['b'], res['s_kv'], res['d_v']]]
                res['out'] = [res['b'] * res['nh_q'], res['s_q'], res['d_v']]
        return res

    def __call__(self, shape, decoding=False, causal=False, **kwargs):
        data = self.get_mkn(shape, decoding, **kwargs)
        scale = 2 if self.bwd else 1
        scale *= 0.5 if causal else 1
        flops_2d_in_fwd = self.gemm_flops(data['s_q'], data['d_qk'], data['s_kv'], data['b'] * data['nh_q'] * scale)
        flops_2d_in_fwd *= 2 if self.bwd else 1
        flops_1d_fwd = self.get_flops(data['b'] * data['nh_q'], data['s_q'], data['s_kv'], self.r['flops']['1D'])
        flops_sfu_fwd = self.get_flops(data['b'] * data['nh_q'], data['s_q'], data['s_kv'], self.r['flops']['SFU'])
        flops_2d_fwd = self.gemm_flops(data['s_q'], data['s_kv'], data['d_v'], data['b'] * data['nh_q'] * scale)
        res = {
            'op_type': 'FA',
            'B': self._int(data['b'] * data['nh_q']),
            'M': self._int(data['s_q']),
            'K': self._int(data['s_kv']),
            'N': self._int(data['d_v']),
            'shape': self._int(shape),
            'flops': {'2D': [flops_2d_in_fwd, flops_2d_fwd], '1D': [flops_1d_fwd], 'SFU': [flops_sfu_fwd]},
            'flops_fwd_bwd': 6 * flops_2d_fwd,
            '2D_flops': flops_2d_in_fwd + flops_2d_fwd,
            'io': self.get_io(data['s_q'], data['s_kv'], data['nh_q'], data['nh_kv'], data['d_qk'], data['d_v'], data['b']),
            'params': self.get_weight(data['s_q'], data['s_kv'], data['d_v']),
            'activation': self.get_activation(data['s_q'], data['s_kv'], data['nh_q'] *  data['d_v'], data['b']),
        }
        return res

    @staticmethod
    def get_flops(b, m, n, r=1):
        return b * m * n * r

    @staticmethod
    def gemm_flops(m, k, n, b=1):
        # 𝐴𝑚×𝑘 ×𝑋𝑘×𝑛 matrix multiplication requires 2𝑚 ×𝑘 ×𝑛 FLOPs
        return 2 * m * k * n * b

    @staticmethod
    def get_io(seqlen_q, seqlen_kv, nheads_q, nheads_kv, headdim_qk, headdim_v, b=1):
        io = 0
        io += b * seqlen_q * nheads_q * headdim_qk * 2 # input_q.numel()
        io += b * seqlen_kv * nheads_kv * (headdim_qk + headdim_v) * 2 # input_kv.numel()
        io += b * seqlen_q * nheads_q * headdim_v * 2 # input_qv.numel() if has_qv
        io += b * seqlen_q * nheads_q * headdim_v * 2 # output.numel()
        return io

    @staticmethod
    def get_weight(m, k, n):
        return 0

    @staticmethod
    def get_activation(m, k, n, b):
        # add expsum and xmax
        return b * m * n + b * m