import vllm.utils
import inspect
import makefun

# 保存原始函数
original_direct_register_custom_op = vllm.utils.direct_register_custom_op
original_sig = inspect.signature(original_direct_register_custom_op)

def patched_direct_register_custom_op(*args, **kwargs):
    bound = original_sig.bind(*args, **kwargs)
    bound.apply_defaults()
    op_name = bound.arguments.get("op_name", None)
    op_func = bound.arguments.get("op_func", None)

    # 用 makefun.wraps 保持签名
    @makefun.wraps(op_func)
    def wrapped_op_func(*a, **k):
        print(f"Calling custom op: {op_name}")
        print(f"Inputs: {[arg.shape if hasattr(arg, 'shape') else arg for arg in a]}")
        print(f"Kwargs: {[v.shape if hasattr(v, 'shape') else v for v in k.values()]}")
        result = op_func(*a, **k)
        print(f"Output: {result.shape if hasattr(result, 'shape') else result}")
        return result

    bound.arguments["op_func"] = wrapped_op_func
    return original_direct_register_custom_op(*bound.args, **bound.kwargs)

# 应用 monkey patch
vllm.utils.direct_register_custom_op = patched_direct_register_custom_op
print("[TorchTrace] vllm_monkey_patch loaded: direct_register_custom_op has been monkey patched.")
