import sys
import time
from torchtrace.utils import get_device_id
from torchtrace.module import tracker

start_time = {}
CFG_TIME_FILTER=False


def filter(frame):
    try:
        caller_file = frame.f_code.co_filename
        if not caller_file:
            return None
        caller_self = frame.f_locals.get("self")
        caller_name = frame.f_code.co_name
        caller_self = caller_self.__class__.__name__ if caller_self is not None \
                and hasattr(caller_self, '__class__') else caller_name
        if caller_self.startswith('<') and caller_self.endswith('>'):
            return None
        # if 'ops' in caller_file and all(k not in caller_file for k in ['loops', '_ops']):
        #     return caller_self
        if any(k in caller_self.lower() for k in ['world', 'group', 'identity', 'hook', '__init__']):
            return None
        if any(caller_file.endswith(f) for f in ['functional.py', 'distributed_c10d.py']):
            if any(k in caller_self.lower() for k in ['_check_single_tensor', 'is_initialized', 'get_rank', '_check_valid_timeout']):
                return None
        if any(k in caller_self for k in ['BackwardHookFunction', 'record_function', '__new__', 'unpack_dual', \
                     'apply', '__instancecheck__', '_tensor_or_tensors_to_tuple', '_make_grads', '_is_setup_context_defined', \
                     '_engine_run_backward', 'grad', '_enter_inference_mode', 'MakeViewlessTensor']):
            return None
            # return caller_self
        if 'custom_op' in caller_file:
            return caller_self
        # if caller_file.endswith('module.py') and caller_name == '_wrapped_call_impl':
        #     return caller_self
        # if any(f in caller_file for f in ['apex', 'transformer_engine', 'flash_attn', 'autograd']) \
        # if any(f in caller_file for f in ['autograd']) \
        #         and all(k not in caller_file for k in ['schemas', 'aot_autograd']):
        #     return caller_self
        # if caller_file.endswith('optimizer.py') and not str(caller_self).startswith('<module>'):
        #     return caller_self
    except:
        pass
    return None


def __torch_api__(func_name, frame, event, return_value):
    device_id = None
    kwargs = {}
    for attr in list(frame.f_locals.keys()):
        if attr in ['self', '__class__', 'cls']:
            continue
        if any(k in attr for k in ['model', 'module']):
            continue
        value = frame.f_locals[attr]
        kwargs[attr] = value
        if not device_id:
            try:
                device_id = get_device_id(value) if value else None
            except:
                pass
    tracker.trace_api(func_name, event, device_id, frame.f_code.co_filename, [], kwargs, return_value)


def trace_function(frame, event, arg):
    global start_time
    if event == 'call':
        func_name = filter(frame)
        if func_name:
            if CFG_TIME_FILTER:
                start_time[func_name] = time.time()
            __torch_api__(func_name, frame, event, arg)
    elif event == 'return':
        func_name = filter(frame)
        if func_name:
            if CFG_TIME_FILTER:
                end_time = time.time()
                if start_time.get(func_name):
                    time_taken = (end_time - start_time[func_name])
                    start_time.pop(func_name)
                else:
                    time_taken = 0.0
                if time_taken > 0.0001:
                    # 不能保证每个 return 时间都会被记录
                    __torch_api__(func_name, frame, event, arg)
            else:
                __torch_api__(func_name, frame, event, arg)
    return trace_function


def api_trace_start():
    sys.settrace(trace_function)


def api_trace_end():
    sys.settrace(None)
