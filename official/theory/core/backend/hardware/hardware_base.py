from .efficiency import Efficiency

class HardwareObject():
    def __init__(self, name, cfg, efficiency_cfg, extra_config) -> None:
        self.name = name
        self._compute = cfg['compute']
        self._storage = cfg['storage']
        self._link = cfg['link']
        self._efficiency = Efficiency(cfg, efficiency_cfg, extra_config)

    def config(self, attr, key, value):
        try:
            setattr(getattr(self, f'_{attr}'), key, value)
        except:
            getattr(self, f'_{attr}')[key] = value

    def print(self):
        print('*'*50, self.name ,'*'*50)
        print(self._compute)

    def get_tflops(self):
        return self._compute['2D_fp16']

    @staticmethod
    def get_flops_utilization(total, n_devices, device_2d, t):
        # 2.51E+18 / 10 / (1536*312E+12) = 52.28%
        return total / (n_devices * device_2d) / t

    @staticmethod
    def get_latency_per_iter(total, n_devices, device_2d, utilization):
        # 2.51E+18 / 10 / (1536*312E+12) = 52.28%
        device_flops = n_devices * device_2d * utilization
        print(n_devices, 'device:', device_flops, 'tflops per s')
        return total / device_flops

    @staticmethod
    def get_peak_flops():
        # peak_flops = F_clk * F_req * N_SM，
        # F_clk 为 GPU 的时钟周期内指令执行数 (单位为 FLOPS/Cycle)， A100 为 64 Flops/Cycle
        # Tensor Core里面的是MAC或者FFA，融合乘加指令，所以一次指令执行会计算2次（一次乘法一次加法），因此会乘以2
        # F_req 为运行频率 (单位为 GHz)，A100 为 1.41GHz
        # N_SM 为 GPU SM 数量 (单位为 Cores)，A100 为 108
        # 约等于 1.95 TFLOPS
        pass

    @staticmethod
    def convert(total, params):
        total = total / 1000000000000
        print('needed:', total / 1000  / params['mini_bs'] * 750, 'pflops')
        n_devices = params['tensor_parallel'] * params['pipeline_parallel']
        if 'data_parallel' in params:
            n_devices *= params['data_parallel']
        flops_per_device = '2D'
        if params['mix'] != 'fp32':
            flops_per_device += '_' + params['mix']
        return total, n_devices, flops_per_device

    def flops_utilization_cluster(self, total, n_devices, precision, t):
        total = total / 1000000000000
        flops_per_device = '2D'
        if precision == 'fp8':
            flops_per_device += '_' + 'fp8'
        elif precision != 'fp32':
            flops_per_device += '_' + 'fp16'
        return self.get_flops_utilization(total, n_devices, self._compute[flops_per_device], t)

    def flops_utilization(self, total, params, t):
        total, n_devices, flops_per_device = self.convert(total, params)
        return self.get_flops_utilization(total, n_devices, self._compute[flops_per_device], t)

    def latency_per_iter(self, total, params, utilization):
        total, n_devices, flops_per_device = self.convert(total, params)
        return self.get_latency_per_iter(total, n_devices, self._compute[flops_per_device], utilization)

    def get_ccl_latency(self, ccl_type, params, data_shape, dtype, T, link_type, duration):
        # nvlink 3: 600 G/s total; 50G/s single
        # pcie switch + rdma: 100 G/s total; 50G/s single
        if isinstance(dtype, str):
            if dtype in ['fp16', 'bf16']:
                total = 2
            elif dtype in ['fp8', 'int8']:
                total = 1
            elif dtype in ['fp32']:
                total = 4
            elif dtype in ['int64']:
                total = 8
        else:
            total = dtype
        if isinstance(data_shape, list):
            for i in data_shape:
                total *= i
        else:
            total *= data_shape
        if duration:
            total /= duration
        elif self._efficiency:
            if self._efficiency[ccl_type]:
                total /= self._efficiency[ccl_type]
            else:
                total /= self._efficiency['D-D']
        else:
            if ccl_type == 'broadcast':
                total *= (T - 1) / 1000000
            else:
                total *= (T - 1)/T / 1000000
        # total /= 1000000
        # bandwidth = 40
        if isinstance(params[link_type], int):
            bandwidth = params[link_type] * 1000
        else:
            bandwidth = min(params[link_type]['single_in_out'], params[link_type]['total']/T)
        latency = total / bandwidth
        # print('ccl data:', total, 'M', bandwidth, 'M/s', latency, 'us')
        return latency

    def compute_DD_in_Node(self, ccl_type, data_shape, bpe, decoding=False, duration=None):
        T = 2
        link_type = 'D-D'
        ccl_type = ccl_type.replace('c10d::', '')
        return self.get_ccl_latency(ccl_type, self._link, data_shape, bpe, T, link_type, duration)

    def compute_2D(self, scale, op_shape, dtype, duration=None, dict_out=True):
        res = {}
        flops = scale
        if isinstance(op_shape, list):
            for shape in op_shape[0]:
                flops *= shape
            for shape in op_shape[1][:-1]:
                flops *= shape
        else:
            flops *= op_shape
        flops /= 1000000
        # res['flops'] = flops / 1000
        _d = duration if duration else self._efficiency['2D_fp16']
        res['f16'] = flops / self._compute['2D_fp16'] / _d
        if '2D_fp8' in self._compute and self._compute['2D_fp8'] != 0:
            _d = duration if duration else self._efficiency['2D_fp8']
            res['fp8']  = flops / self._compute['2D_fp8'] / _d
        res['base'] = res['fp8'] if dtype in ['fp8', 'int8', 'f8', 'i8'] and 'fp8' in res else res['f16']
        return res if dict_out and not duration else res['base']

    def is_fp8(self, dtype):
        return '2D_fp8' in self._compute and self._compute['2D_fp8'] != 0 and dtype in ['fp8', 'int8', 'f8', 'i8']

    def compute_1D(self, scale, op_shape, typ='1D', duration=None, dict_out=True):
        flops = scale
        if isinstance(op_shape, list):
            for shape in op_shape[0]:
                flops *= shape
        else:
            flops *= op_shape
        flops /= 1000000
        _d = duration if duration else self._efficiency[typ]
        res = {'base': flops / self._compute[typ] / _d}
        return res if dict_out and not duration else res['base']

    def compute_L3_BW(self, scale, op_shape, decoding, duration=None):
        io = scale
        if isinstance(op_shape, list):
            for shape in op_shape[0]:
                io *= shape
        else:
            io *= op_shape
        io /= 1000
        _efficiency = self._efficiency['L3_bandwidth_s'] if decoding and 'L3_bandwidth_s' in self._efficiency._efficiency else self._efficiency['L3_bandwidth']
        _d = duration if duration else _efficiency
        return io / self._storage['L3_bandwidth'] / _d

    def exec(self, op, op_type, data, dtype='fp16', decoding=False, duration=None):
        latency_compute = 0
        latency_io = 0
        self._efficiency.set(op_type, data, self.bpe(dtype))
        if dtype in data and data['dtype']:
            dtype = data['dtype']
        if not duration is None:
            duration -= self._efficiency['launch']
            if duration <= 0:
                duration = 0.1
        if data.get('flops'):
            latency_compute = self.compute(op, op_type, data['flops'], dtype, scale=1, duration=duration, dict_out=False)
        if data.get('io'):
            latency_io = self.get_io(op, op_type, data['io'], self.bpe(dtype), decoding, duration)
        if not duration is None:
            if not isinstance(latency_compute, dict) and latency_compute > 1000:
                latency_compute = 100
            if not isinstance(latency_io, dict) and latency_io > 1000:
                latency_io = 100
            if isinstance(latency_compute, dict):
                latency_compute['sfu_scale'] = self._compute['1D'] / self._compute['SFU']
            return {'flops_utilization': latency_compute, 'io_utilization': latency_io}
        return max(latency_compute, latency_io) + self._efficiency['launch']

    @staticmethod
    def bpe(dtype):
        if dtype in ['fp16', 'bf16', 'f16']:
            bpe = 2
        elif dtype in ['fp8', 'int8', 'f8', 'i8']:
            bpe = 1
        elif dtype in ['fp32', 'f32', 'i32']:
            bpe = 4
        elif dtype in ['int64', 'i64']:
            bpe = 8
        return bpe

    def get_io(self, op, op_type, op_shape, bpe, decoding, duration=None):
        if op_type == 'Comm':
            return self.compute_DD_in_Node(op, op_shape, bpe, decoding, duration)
        else:
            return self.compute_L3_BW(bpe, op_shape, decoding, duration)

    def compute(self, op, op_type, op_shape, dtype, scale=1, duration=None, dict_out=True):
        if op_type == 'FA':
            if isinstance(scale, list):
                print('Err, FA not support directly compute')
                exit()
            elif isinstance(op_shape, list):
                print('Err, FA not support directly compute')
                exit()
            elif isinstance(op_shape, dict):
                res = {}
                for run_type in op_shape:
                    this_run_scale = scale
                    if run_type == '2D':
                        if self.is_fp8(dtype):
                            this_run_scale *= self._efficiency['2D_fp8'] / self._efficiency['FA_fp16']
                        else:
                            this_run_scale *= self._efficiency['2D_fp16'] / self._efficiency['FA_fp16']
                    else:
                        this_run_scale *= self._efficiency[run_type] / self._efficiency['FA_1D']
                    _shape = sum(op_shape[run_type])
                    res[run_type] = self.compute(op, run_type, _shape, dtype, this_run_scale, duration, dict_out=False)
                if duration:
                    # return {'2D': res['2D'], '1D&SFU': res['1D'] + res['SFU']}
                    return res
                split_list = []
                split_list.append(max(res['2D'], res['1D'] + res['SFU']))
                if self._efficiency['FA_parallel'] > 1:
                    split_list.append(max(res['2D'] + res['1D'], res['SFU']))
                    split_list.append(max(res['2D'] + res['SFU'], res['1D']))
                if self._efficiency['FA_parallel'] > 2:
                    split_list.append(max(res['2D'], res['1D'], res['SFU']))
                res = {'base': min(split_list)}
                return res if dict_out else res['base']
            print('Err, FA not support this shape input')
            exit()
        elif op_type == '2D':
            if dtype in ['fp8', 'f8'] and self._compute.get('per_group_fp8_by_sw'):
                res1D = self.compute_1D(scale, op_shape/128, '1D', duration, dict_out=dict_out)
                res2D = self.compute_2D(scale, op_shape, dtype, duration, dict_out=dict_out)
                return res1D + res2D
            return self.compute_2D(scale, op_shape, dtype, duration, dict_out=dict_out)
        elif op_type in ['1D', 'RNG', 'SFU']:
            if isinstance(op_shape, dict):
                res = {}
                for run_type in op_shape:
                    _shape = sum(op_shape[run_type])
                    res[run_type] = self.compute(op, run_type, _shape, dtype, scale, duration, dict_out=False)
                latency = sum(res.values())
                if not duration:
                    res['base'] = latency
                # else:
                #     base = '&'.join(res.keys())
                #     res = {base: latency}
                return res if dict_out or duration else res['base']
            else:
                return self.compute_1D(scale, op_shape, op_type, duration, dict_out=dict_out)
        else:
            return {}

    def calibration(self, method, case, x_data, y_data, calibration):
        return self._efficiency.calibration(method, case, x_data, y_data, calibration)

    def verification(self, method, case, x_data, y_data, calibration):
        return self._efficiency.verification(method, case, x_data, y_data, calibration)