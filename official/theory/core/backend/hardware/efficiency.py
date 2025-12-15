import numpy as np
from scipy.optimize import curve_fit
import os
import math


# 定义非对称 S 型曲线函数
def asymmetric_sigmoid(x, max, grad, middle):
    # launch time: min
    # middle efficiency: middle
    # grad: grad
    # max efficiency: max
    return max / (1.0 + np.exp(-grad * (x - middle))) + 1

def default_func(x, launch):
    return launch

CURVE = {
    "asymmetric_sigmoid": asymmetric_sigmoid,
    "std": default_func,
}


class Efficiency():
    def __init__(self, cfg, efficiency_cfg, extra_cfg) -> None:
        self._efficiency = {'2D_fp16': 0.9,
                           'Batch_Dot_fp16': 0.9,
                           'FA_fp16': 0.8,
                           'FA_1D': 0.5,
                           'FA_parallel': 1,
                           '2D_fp8': 0.9,
                           '1D': 0.9,
                           'launch': 2,
                           'L3': 0.9,
                           'L3_bandwidth': 0.8,
                           'D-D': 0.8,
                           'North_bandwidth': 0.6}
        self.popt = efficiency_cfg if efficiency_cfg else {}
        self.popt.update(extra_cfg)
        if cfg.get('efficiency'):
            for k, v in cfg['efficiency'].items():
                self._efficiency[k] = v
        self._efficiency['SFU'] = self._efficiency['1D']
        self.popt_default = {'2D': {'asymmetric_sigmoid': [90, 1.2, 21]},
                             'base': {'asymmetric_sigmoid': [90, 1.2, 13]}}
        self.upper_bound = True
        self.op_type = ''
        self.data = ''
        self.bpe = 1

    def __getitem__(self, key):
        if not self.upper_bound:
            if key in ['L3_bandwidth', 'L3_bandwidth_s', 'North_bandwidth', 'D-D', 'H-D']:
                return self.get_efficiency(key, method='asymmetric_sigmoid', type='io')
            if key in ['2D_fp16', '2D_fp8']:
                return self.get_efficiency(key, method='asymmetric_sigmoid', type='flops')
            if key == 'launch':
                return self.get_efficiency(key, method='std')
            if key in self.popt:
                return self.get_efficiency(key, method='asymmetric_sigmoid', type='io', scale=self.bpe)
        if key in self._efficiency:
            return self._efficiency[key]

    def __setitem__(self, key, value):
        if key in self._efficiency:
            self._efficiency[key] = value

    def set(self, op_type, data, bpe):
        self.op_type = op_type
        self.data = data
        self.bpe = bpe

    def get_popt(self, key, method):
        default = 'base'
        for cls in self.popt_default:
            if cls in key:
                default = cls
                break
        return self.popt[key][method] if key in self.popt else self.popt_default[default][method]

    def calibration(self, method, case, x_data, y_data, calibration):
        # 拟合曲线 color='red'
        popt_default = self.get_popt(calibration, method)
        # 拟合曲线
        popt, pcov = curve_fit(CURVE[method],
                                x_data,
                                y_data,
                                p0=popt_default, maxfev=50000)
        y_line = CURVE[method](x_data, *popt)

        # 打印拟合参数
        # print("拟合参数:", case, popt)
        return popt, y_line

    def verification(self, method, case, x_data, y_data, calibration):
        # 拟合曲线 color='yellow'
        popt_default = self.get_popt(calibration, method)
        y_line = CURVE[method](x_data, *popt_default)

        return self.popt[calibration][method], y_line

    def get_efficiency(self, key, method, type='flops', scale=1):
        efficiency_scale = 1
        if key in ['launch']:
            data = 0
        elif isinstance(self.data[type], dict):
            data = self.data[type]['2D'][0] * scale
        else:
            if '2D' in key and 'B' in self.data and self.data['B'] > 1:
                efficiency_scale = self._efficiency['Batch_Dot_fp16'] / self._efficiency['2D_fp16']
            data = self.data[type] * scale
        if key in self.popt and method in self.popt[key]:
            x_data = math.log(data) if data > 0 else 0
            efficiency = CURVE[method](x_data, *self.popt[key][method])
            if key in ['launch']:
                return efficiency
            if efficiency < 5:
                efficiency = 5.0
            return efficiency / 100 * efficiency_scale
        return self._efficiency[key]