import os
from omegaconf import OmegaConf, DictConfig
from .hardware_base import HardwareObject
from theory.core.utils import backend_path

OmegaConf.register_new_resolver("multiply", lambda x, y: x * y, replace=True)

class HardwareLoader():
    def __init__(self):
        self.base_config_path = os.path.join(backend_path(), 'hardware/conf')
        self.hw_objs = {}
        self.load()

    def load(self):
        config = OmegaConf.load(os.path.join(self.base_config_path, 'device.yaml'))
        OmegaConf.resolve(config)
        config_dict = OmegaConf.to_container(config, resolve=True)
        efficiency = OmegaConf.load(os.path.join(self.base_config_path, 'efficiency.yaml'))
        OmegaConf.resolve(efficiency)
        efficiency_dict = OmegaConf.to_container(efficiency, resolve=True)
        for hw_name, hw_config in config_dict.items():
            efficiency_config = efficiency_dict.get(hw_name)
            extra_config = efficiency_dict.get('OP')
            hw_obj = HardwareObject(hw_name, hw_config, efficiency_config, extra_config)
            self.hw_objs[hw_name] = hw_obj
        return config_dict

    def get(self, hw_name):
        return self.hw_objs[hw_name]


hw_loader = HardwareLoader()