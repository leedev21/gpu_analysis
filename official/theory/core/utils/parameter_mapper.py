"""
Parameter mapping utilities for consistent naming across the system.
This module provides functions to map external parameter names to internal names.
"""

# Centralized mapping dictionary
# External name (key) -> internal name (value)
PARAMETER_MAPPING = {
    "hardware": "hw",
    "software": "sw",
    "world_size": "n_device",
    "local_rank": "device",
    "train_iters": "max_steps",
    "num_experts": "n_experts",
    "moe_router_num_groups": "n_expert_groups",
    "parameters_size": "size",
    "total_time": "total_latency",
    "training_time_per_epoch": "train_epoch_timing",
    "training_time_total": "total_time",
    "training_time_per_iter": "train_iter_timing",
}

def map_parameter_name(name):
    """Map a single parameter name from external to internal representation.

    Args:
        name: The external parameter name

    Returns:
        The internal parameter name
    """
    return PARAMETER_MAPPING.get(name, name)

def map_config_dict(config):
    """递归映射配置字典中的所有参数名（无论嵌套层级）

    Args:
        config: 配置字典

    Returns:
        映射后的字典
    """
    if not isinstance(config, dict):
        return config

    result = {}
    for key, value in config.items():
        # 对每个键都应用映射
        mapped_key = map_parameter_name(key)

        # 递归处理嵌套结构
        if isinstance(value, dict):
            result[mapped_key] = map_config_dict(value)
        elif isinstance(value, list):
            result[mapped_key] = [
                map_config_dict(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[mapped_key] = value

    return result