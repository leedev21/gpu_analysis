class ModelPerformanceConfig():
    def __init__(self) -> None:
        self.args = {
                'HW': ('hardware', 'A100'),
            }
        self.duration = {
                'total': ('total_time', 0.0),
                'step': ('step_avg_time', 0.0),
                'iter': ('iter_avg_time', 0.0),
            }
        self.flops_tps = {
                '2D_flops': ('flops_2D', 0.0),
                'MFU': ('MFU', 0.0),
                'tps': ('tokens_per_sec_per_device', 0.0),
                'flops': ('device_tflops', 0.0),
                'io': ('io', 0.0),
                '2D_flops': ('flops_2D', 0.0),
            }
        self.memory = {
                'params': ('weight', 0.0),
                'activation': ('forward_activation', 0.0),
                'optim': ('optimizer', 0.0),
                'total_mem': ('total_memory', 0.0),
            }
        self.bubble_overlap = {
                'bubble': ('bubble', 0.0),
                'bubble_rate': ('bubble_rate', 0.0),
            }
        self.percentage = {
                'default_2D_percent': ('default_percentage_2D', 0.0),
                'decoding_2D_percent': ('decoding_percentage_2D', 0.0),
                'total_2D_percent': ('total_percentage_2D', 0.0),
                'default_FA_percent': ('default_percentage_FA', 0.0),
                'decoding_FA_percent': ('decoding_percentage_FA', 0.0),
                'total_FA_percent': ('total_percentage_FA', 0.0),
                'default_comm_percent': ('default_percentage_COMM', 0.0),
                'decoding_comm_percent': ('decoding_percentage_COMM', 0.0),
                'total_comm_percent': ('total_percentage_COMM', 0.0),
            }


class TrainingPerformanceConfig(ModelPerformanceConfig):
    def __init__(self) -> None:
        super().__init__()
        self.percentage = {
                'total_2D_percent': ('total_percentage_2D', 0.0),
                'total_FA_percent': ('total_percentage_FA', 0.0),
                'total_comm_percent': ('total_percentage_COMM', 0.0),
                }


class InferencePerformanceConfig(ModelPerformanceConfig):
    def __init__(self) -> None:
        super().__init__()
        self.percentage = {
                'default_2D_percent': ('prefill_percentage_2D', 0.0),
                'decoding_2D_percent': ('decoding_percentage_2D', 0.0),
                'total_2D_percent': ('total_percentage_2D', 0.0),
                'default_FA_percent': ('prefill_percentage_FA', 0.0),
                'decoding_FA_percent': ('decoding_percentage_FA', 0.0),
                'total_FA_percent': ('total_percentage_FA', 0.0),
                'default_comm_percent': ('prefill_percentage_COMM', 0.0),
                'decoding_comm_percent': ('decoding_percentage_COMM', 0.0),
                'total_comm_percent': ('total_percentage_COMM', 0.0),
                }


class ModelPerformance():
    def __init__(self, config) -> None:
        self.args = {}
        self.duration = {}
        self.flops_tps = {}
        self.memory = {}
        self.bubble_overlap = {}
        self.percentage = {}
        self.init_with_config(config)

    def init_with_config(self, config):
        for attr in ['args', 'duration', 'flops_tps', 'memory', 'bubble_overlap', 'percentage']:
            if hasattr(config, attr) and hasattr(self, attr):
                for k, items in getattr(config, attr).items():
                    getattr(self, attr)[k] = items[1]

    def to_str(self, value):
        if tag == 'time':
            return TimeObj(value, unit='ns').to_str()
        if tag = '%':
            return str(value * 100, 3)+'%'

    def add(self) -> None:
        pass