"""
TorchTrace plugin for vLLM integration.

This module provides the plugin entry point for vLLM to initialize TorchTrace
tracing capabilities when loaded as a plugin.
"""

import os


def initialize_torchtrace() -> None:
    """
    Initialize TorchTrace plugin for vLLM.

    This function is called by vLLM's plugin system when the torchtrace plugin
    is loaded. It sets up TorchTrace with appropriate default configurations
    for vLLM workloads.
    """

    try:
        import vllm.envs as envs
    except ImportError:
        print("[TorchTrace] vLLM is not installed")
        return

    allowed_plugins = envs.VLLM_PLUGINS
    if allowed_plugins is None or "torchtrace" not in allowed_plugins:
        print("[TorchTrace] TorchTrace plugin is not allowed to be loaded")
        return

    try:
        import torchtrace.vllm_monkey_patch #noqa

        from torchtrace import set_torchtrace, update

        print("[TorchTrace] Initializing TorchTrace plugin for vLLM")

        # ═══════════════════════════════════════════════════════════════
        # 环境变量配置（可以同时使用 customer_op 和 filter）
        # ═══════════════════════════════════════════════════════════════
        # TORCHTRACE_PLUGIN_CUSTOMER_OP: customer_op 配置（字典字符串）
        #   示例: "{'apex':'','vllm':'','flash_attn':'flash_attn_device'}"
        #   留空则使用默认配置
        #
        # TORCHTRACE_PLUGIN_FILTER: filter 配置（逗号分隔）
        #   示例: "op_trace,{op:[aten::add,aten::mul]}"
        #   示例: "op_trace,default"
        #   留空则不启用 filter
        # ═══════════════════════════════════════════════════════════════

        customer_op_config = os.getenv("TORCHTRACE_PLUGIN_CUSTOMER_OP", "")
        filter_config = os.getenv("TORCHTRACE_PLUGIN_FILTER", "")

        # ────────────────────────────────────────────────────────────
        # 1. 配置 customer_op（hook 自定义算子）
        # ────────────────────────────────────────────────────────────
        if customer_op_config:
            # 用户提供了自定义配置
            try:
                import ast
                customer_op_dict = ast.literal_eval(customer_op_config)
                update('customer_op', customer_op_dict)
                print(f"[TorchTrace] ✓ Enabled customer_op with config: {customer_op_dict}")
            except Exception as e:
                print(f"[TorchTrace] ✗ Failed to parse TORCHTRACE_PLUGIN_CUSTOMER_OP: {e}")
                print("[TorchTrace] → Using default customer_op config")
                update('customer_op')
        else:
            # 使用默认配置
            update('customer_op')
            print("[TorchTrace] ✓ Enabled customer_op with default config")

        # ────────────────────────────────────────────────────────────
        # 2. 配置 filter（过滤追踪的算子/模块）
        # ────────────────────────────────────────────────────────────
        if filter_config:
            if ',' in filter_config:
                param1, param2 = filter_config.split(',', 1)
                param1 = param1.strip()
                param2 = param2.strip()
                update('filter', param1, param2)
                print(f"[TorchTrace] ✓ Enabled filter: {param1}, {param2}")
            else:
                print(f"[TorchTrace] ✗ Invalid filter config (missing comma): {filter_config}")
                print("[TorchTrace] → Filter not enabled")
        else:
            print("[TorchTrace] ○ Filter not configured (will trace all operations)")

        # Get configuration from environment variables
        torch_dispatch_trace = _get_env_bool("TORCHTRACE_PLUGIN_DISPATCH", True)
        torch_api_trace = _get_env_bool("TORCHTRACE_PLUGIN_API", False)
        save_pt = _get_env_bool("TORCHTRACE_PLUGIN_SAVE_PT", False)
        sync_mode = _get_env_bool("TORCHTRACE_PLUGIN_SYNC", True)
        save_to = os.getenv("TORCHTRACE_PLUGIN_SAVE_TO", "")

        # Initialize TorchTrace with configuration
        set_torchtrace(
            torch_dispatch_trace=torch_dispatch_trace,
            torch_api_trace=torch_api_trace,
            save_pt=save_pt,
            sync_mode=sync_mode,
            save_to=save_to
        )

        print("[TorchTrace] Plugin initialized successfully")
        print("[TorchTrace] Configuration:")
        print(f"[TorchTrace]   - customer_op_config: {customer_op_config or 'default'}")
        print(f"[TorchTrace]   - filter_config: {filter_config or 'not configured'}")
        print(f"[TorchTrace]   - torch_dispatch_trace: {torch_dispatch_trace}")
        print(f"[TorchTrace]   - torch_api_trace: {torch_api_trace}")
        print(f"[TorchTrace]   - save_pt: {save_pt}")
        print(f"[TorchTrace]   - sync_mode: {sync_mode}")
        print(f"[TorchTrace]   - save_to: {save_to or 'default'}")

    except ImportError as e:
        print(f"[TorchTrace] Error: Failed to import TorchTrace: {e}")
        print("[TorchTrace] Make sure TorchTrace is properly installed")
        raise
    except Exception as e:
        print(f"[TorchTrace] Error: Failed to initialize TorchTrace plugin: {e}")
        raise


def _get_env_bool(env_var: str, default: bool) -> bool:
    """
    Get boolean value from environment variable.

    Args:
        env_var: Environment variable name
        default: Default value if environment variable is not set

    Returns:
        Boolean value parsed from environment variable or default
    """
    value = os.getenv(env_var)
    if value is None:
        return default

    return value.lower() in ('true', '1', 'yes', 'on')


def get_plugin_info() -> dict:
    """
    Get information about the TorchTrace plugin.

    Returns:
        Dictionary containing plugin information
    """
    return {
        'name': 'torchtrace',
        'version': '0.1',
        'description': 'TorchTrace integration plugin for vLLM',
        'author': 'jimmy.qin',
        'supported_features': [
            'torch_dispatch_tracing',
            'torch_api_tracing',
            'custom_operation_tracing',
            'performance_profiling'
        ]
    }