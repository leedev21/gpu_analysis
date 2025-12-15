from typing import List, Union, Tuple

from torch.utils._python_dispatch import (
    _get_current_dispatch_mode_stack,
    TorchDispatchMode,
)
from torch.overrides import TorchFunctionMode

from torchtrace.utils import (
    get_device_id,
    GLOBAL_CFG
)
from torchtrace.module import tracker
import atexit
import contextlib


class MyTorchDispatchMode(TorchDispatchMode):

    def __init__(self):

        # True if we enter'ed and actually enabled fake tensor mode,
        # false if it was a no-op.  Not thread safe but neither is
        # in_kernel_invocation
        self.enter_stack: List[bool] = []
        self.save = False
        self.call_depth = 0
        self.skip = False

    # Without this override, running torch.compile under DebugMode
    # will force torch.compile to always use the “eager” backend
    # With this, DebugMode will not take effect on torch.compile
    @classmethod
    def ignore_compile_internals(cls):
        return True

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # print(' '*self.call_depth, func)
        try:
            self.call_depth += 1
            return func(*args, **kwargs)
        finally:
            self.call_depth -= 1

    @contextlib.contextmanager
    def record_redistribute_calls(self, arg_idx, src_placement, dst_placement):
        try:
            args = (
                [arg_idx, transform_info_str]
                if transform_info_str
                else [arg_idx, src_placement, dst_placement]
            )
            func = 'redistribute_input'
            kwargs = {}
            self.call_depth += 1
            print(' '*self.call_depth, func, args, kwargs)
            yield
        finally:
            self.call_depth -= 1

    def op_filter(self, op_name):
        if self.skip:
            return True, False
        need_skip = op_name in tracker.support_by_torch
        need_save = self.save and not need_skip

        if GLOBAL_CFG.get('op_trace'):
            try:
                need_save = need_save and op_name in GLOBAL_CFG['op_trace']['op'] \
                        and (not tracker.ops.get(op_name) or GLOBAL_CFG['op_trace']['max'] < 1 or \
                        len(tracker.op_info.get(op_name, {})) < GLOBAL_CFG['op_trace']['max'])
                # print('need_save:', need_save, op_name, tracker.ops.get(op_name), GLOBAL_CFG['op_trace'], len(tracker.op_info[op_name]))
                if not need_save:
                    return True, False
            except Exception as e:
                print(f"Filter error for {op_name}: {e}")
                raise
        elif GLOBAL_CFG['skip_mode']:
            if (need_skip and not tracker.support_by_torch[op_name]) or tracker.state == 'start':
                return need_skip, False

        return False, need_save

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert self not in _get_current_dispatch_mode_stack(), func

        need_skip, need_save = self.op_filter(func._schema.name)
        if need_skip:
            return func(*args, **kwargs)
        if any(k == func._schema.name for k in ['aten::set_']):
            return func(*args, **kwargs)

        default_args = [] # 确定使用列表存储 default_args
        default_kwargs = {} #使用字典存储 default_kwargs
        for i, arg in enumerate(func._schema.arguments):
            if arg.kwarg_only:
                default_kwargs[arg.name] = arg.default_value
            else:
                default_args.append((arg.name, arg.default_value))
        
        # 获取真实的schema信息
        real_schema = tracker._get_schema_from_torch_dispatch(func)
        
        try:
            # 获取当前时间
            device_id = get_device_id(args) if args else None
            cache = tracker.trace_op(func._schema.name, 'call', device_id, need_save, args, kwargs, default_args, default_kwargs, real_schema=real_schema)
            # print(f"{formatted_time} dispatch_call: {OPTRACE_VERSION}{pid} torch.{pt_version}.{func._schema.name}({args_str}{kwargs_str}) \n", end='', flush=True)

            outs = func(*args, **kwargs)

            device_id_out = get_device_id(outs)
            if device_id_out:
                device_id = device_id_out
            
            if outs is None:
                print('outputs None', func._schema.name)
                exit()

            # 如果outs_str第一个字符不是 % ，optrace暂时不支持这类语法，解析的时候需要跳过。
            # 例如：A2151:2:0 ScriptObject____torch___torch_classes_c10d_Work_:ScriptObject = torch.2_3_0.c10d::barrier(%5:<1xu8>{1}, ScriptObject____torch___torch_classes_c10d_ProcessGroup_:ScriptObject, %6:list{}, -1:int)

            tracker.trace_op(func._schema.name, 'return', device_id, need_save, outs=outs, cache=cache, real_schema=real_schema)
            # print(f"{formatted_time} dispatch_return: {OPTRACE_VERSION}{pid}:{device_id}:{stream} {outs_str} = torch.{pt_version}.{func._schema.name}({args_str}{kwargs_str}) \n", end='', flush=True)
            return outs

        except TypeError:
            raise

    # No-op if FakeTensorMode is already on the stack
    def __enter__(self, save=False):
        self.save = save
        if self not in _get_current_dispatch_mode_stack():
            self.enter_stack.append(True)
            return super().__enter__()
        else:
            # no-op
            self.enter_stack.append(False)
            return self

    def __exit__(self, a, b, c):
        if len(self.enter_stack) >= 1:
            live = self.enter_stack.pop()
            if live:
                return super().__exit__(a, b, c)


class MyTorchFunctionMode(TorchFunctionMode):
    def __init__(self):
        self.enter_stack: List[bool] = []
        self.save = False
        self.call_depth = 0

    def __enter__(self, save=False):
        self.save = save
        if self not in _get_current_dispatch_mode_stack():
            self.enter_stack.append(True)
            return super().__enter__()
        else:
            # no-op
            self.enter_stack.append(True)
            return self

    def __exit__(self, a, b, c):
        if len(self.enter_stack) >= 1:
            live = self.enter_stack.pop()
            if live:
                return super().__exit__(a, b, c)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # print('MyTorchFunctionMode:', func)
        return func(*args, **kwargs)


myTorchDispatchMode = MyTorchDispatchMode()
# myTorchFunctionMode = MyTorchFunctionMode()

def dispatch_trace_start(save):
    myTorchDispatchMode.__enter__(save)
    # myTorchFunctionMode.__enter__(save)

@atexit.register
def dispatch_trace_end():
    tracker.summary(myTorchDispatchMode.save)
    myTorchDispatchMode.__exit__(None, None, None)
    # myTorchFunctionMode.__exit__(None, None, None)
    tracker.stop_fallback_gpu()
