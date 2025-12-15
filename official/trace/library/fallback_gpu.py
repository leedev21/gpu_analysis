import io
import time
import os
import re
from typing import Union
from queue import Queue
from threading import Thread
from dataclasses import dataclass
from collections import namedtuple
import torch
from torchtrace.library.kv_cache import customer_kv_cache, get_input
try:
    from twisted.internet import reactor, protocol, endpoints
    from twisted.protocols.basic import FileSender
    TWISTED=True
except:
    TWISTED=False

import logging
logging.basicConfig(level=logging.DEBUG)

ApiData = namedtuple('ApiData', ['name', 'args', 'kwargs', 'result', 'step', 'rank', 'time'],
                     defaults=['unknown', None, None, None, 0, 0, 0])

logger = logging.getLogger(__name__)


def move2target_device_(data, target_device):
    return tuple(x.to(target_device) if isinstance(x, torch.Tensor) else x for x in data)


def move2device_exec(obj, device):
    if isinstance(obj, (tuple, list)):
        data_list = [move2device_exec(val, device) for val in obj]
        return data_list if isinstance(obj, list) else tuple(data_list)
    if isinstance(obj, dict):
        return {key: move2device_exec(val, device) for key, val in obj.items()}
    elif isinstance(obj, torch.Tensor):
        obj = obj.detach()
        if obj.device.type != device:
            obj = obj.to(device)
        return obj
    elif "return_types" in str(type(obj)):
        return move2device_exec(tuple(obj), device)
    elif isinstance(obj, torch._C.device):
        return torch.device(device)
    else:
        return obj


def move2target_device(buffer: ApiData, target_device):
    # handle args
    new_args = move2device_exec(buffer.args, target_device)

    # handle kwargs
    new_kwargs = move2device_exec(buffer.kwargs, target_device)

    # handle result
    new_results = move2device_exec(buffer.result, target_device)

    if target_device == torch.device('cpu') or target_device == "cpu":
        return ApiData(buffer.name, tuple(new_args), new_kwargs, new_results, buffer.step, buffer.rank, buffer.time)
    else:
        return ApiData(buffer.name, tuple(new_args), new_kwargs, buffer.result, buffer.step, buffer.rank, buffer.time)


def save_api_data(api_data):
    """Save data to io stream"""
    try:
        io_buff = io.BytesIO()
        torch.save(api_data, io_buff)
    except Exception as e:
        raise RuntimeError(f"save api_data to io_buff failed") from e
    return io_buff


def load_api_data(api_data_bytes):
    """Load data from bytes stream"""
    # try:
    #     buffer = io.BytesIO(api_data_bytes)
    #     buffer = torch.load(buffer, map_location="cpu")
    # except Exception as e:
    #     raise RuntimeError(f"load api_data from bytes failed") from e
    buffer = io.BytesIO(api_data_bytes)
    buffer.seek(0)  # 重置指针到起始位置
    buffer = torch.load(buffer, map_location="cpu", weights_only=False)
    # print("===> buffer: ", buffer.name)
    return buffer


@dataclass
class ATTLConfig:
    is_benchmark_device: bool
    connect_ip: str
    connect_port: int
    # storage_config
    nfs_path: str = None
    tls_path: str = None
    check_sum: bool = True
    queue_size: int = 50


class DebugServer():
    def __init__(self, *args, **kwargs):
        self.signal_exit = False
        self.func = None
        self.out = None
        self.seq_num = -1
        self.device = 'cpu'

    def add_to_sending_queue(self, data, rank=0, step=1):
        if data.step != 0:
            customer_kv_cache.add_rsv(data)
            k_from_cache, v_from_cache = customer_kv_cache.get_kv(data.rank, data.step)
            args, kwargs = get_input(k_from_cache, v_from_cache, data.args, data.kwargs)
        else:
            args, kwargs = data.args, data.kwargs
        args = move2device_exec(args, self.device)
        kwargs = move2device_exec(kwargs, self.device)
        out = self.func(*args, **kwargs)
        self.out = ApiData(
            name=data.name,
            args=data.args,
            kwargs=data.kwargs,
            result=out,
            step=data.step,
            rank=data.rank,
            time=time.time()
        )
        self.seq_num += 1
        return self.seq_num

    def set_device(self, device):
        self.device = device

    def get_resp_wait(self, seq_num, timeout=-1):
        return self.out

    def set_debug_func(self, func):
        self.func = func

    def stop(self):
        pass

    def send_stop_signal(self):
        pass


class ATTL:
    def __init__(self, session_id: str, session_config: ATTLConfig, need_dump=True) -> None:
        self.session_id = session_id
        self.session_config = session_config
        self.logger = logger
        self.socket_manager = None
        self.data_queue = Queue(maxsize=50)
        self.dequeue_list = []
        self.message_end = False
        self.kill_progress = False
        self.check_attl_config()
        self.nfs_path = None
        if self.session_config.nfs_path:
            logger.info(f'start nfs: {self.session_config.nfs_path}')
            self.nfs_path = self.session_config.nfs_path
        elif self.session_config.is_benchmark_device and TWISTED:
            logger.info(f'start socket_manager TCPServer: {self.session_config.connect_port}')
            self.socket_manager = TCPServer(self.session_config.connect_port,
                                            self.data_queue,
                                            self.session_config.check_sum,
                                            self.session_config.tls_path,
                                            send_to_client=True)
            self.socket_manager.start()
        elif need_dump and TWISTED:
            logger.info(f'start socket_manager TCPClient: {self.session_config.connect_ip}')
            from .tcp_client import TCPClient
            self.socket_manager = TCPClient(self.session_config.connect_ip,
                                            self.session_config.connect_port,
                                            self.session_config.check_sum,
                                            self.session_config.tls_path)
            self.socket_manager.start()
            logger.info(f'link to TCPClient: {self.session_config.connect_ip} {self.session_config.connect_port}')
        else:
            logger.info(f'start DebugServer: {DebugServer}')
            self.socket_manager = DebugServer()

    def check_attl_config(self):
        if self.session_config.nfs_path:
            if os.path.exists(self.session_config.nfs_path):
                return
            else:
                raise Exception(f"nfs path {self.session_config.nfs_path} doesn't exists.")
        ipv4_pattern = "([1-9]?\d|1\d{2}|2[0-4]\d|25[0-5])(\.([1-9]?\d|1\d{2}|2[0-4]\d|25[0-5])){3}$"
        if not re.match(ipv4_pattern, self.session_config.connect_ip):
            raise Exception(f"host {self.session_config.connect_ip} is invalid.")
        if not (0 < self.session_config.connect_port <= 65535):
            raise Exception(f"port {self.session_config.connect_port} is invalid.")

    def stop_serve(self):
        if isinstance(self.socket_manager, TCPServer):
            self.socket_manager.stop()

    def send(self, buffer) -> None:
        """
        npu major in 'send' (client)
        """

        # if tcp connection lost,
        if self.socket_manager.signal_exit:
            raise ConnectionError(f"Failed to connect to {self.session_config.connect_ip}.")

        # know receiver receive and go next
        if isinstance(buffer, ApiData):
            buffer = move2target_device(buffer, torch.device('cpu'))

        if 'device' in buffer.kwargs:
            buffer.kwargs.pop('device')
        rank = buffer.rank if hasattr(buffer, "rank") and buffer.rank is not None else 0
        step = buffer.step if hasattr(buffer, "step") else 0
        if isinstance(self.socket_manager, DebugServer):
            data = buffer
        else:
            try:
                io_buff = save_api_data(buffer)
            except Exception as e:
                self.logger.info(f"{buffer.name} can not be saved, skip: {e}")
                return
            data = io_buff.getvalue()
        return self.socket_manager.add_to_sending_queue(data, rank=rank, step=step)

    def send_to_client(self, send_data:ApiData, seq_num, proto_id) -> None:
        buffer = move2target_device(send_data, torch.device('cpu'))
        if 'device' in buffer.kwargs:
            buffer.kwargs.pop('device')
        try:
            # io_buff = save_api_data(buffer.result)
            io_buff = save_api_data(buffer)
        except Exception as e:
            self.logger.info(f"{buffer.name} can not be saved, skip: {e}")
            return
        rank = send_data.rank
        step = send_data.step
        self.socket_manager.add_to_sending_client_queue(io_buff.getvalue(),
                                                        seq_num,
                                                        proto_id,
                                                        rank=rank,
                                                        step=step)

    def recv(self, timeout_ms=0):
        buffer = None
        seq_num = None
        proto_id = None
        while buffer is None:
            if timeout_ms > 0:
                time.sleep(timeout_ms / 1000.0)
            if buffer is None and not self.data_queue.empty():
                result = self.data_queue.get()
                if len(result) == 3:
                    buffer, seq_num, proto_id = result
                else:
                    buffer = result
                break
            if buffer is None and timeout_ms > 0:  # timeout is the only case we give up and return None
                break
            if self.message_end and self.data_queue.empty():
                buffer = b"KILL_CONFIRM"
                self.kill_progress = True
                break
            time.sleep(0.1)  # waiting outside the lock before next attempt
        if buffer is None:
            # this is a result of a timeout
            self.logger.info(f"RECEIVE API DATA TIMED OUT")
        else:
            if buffer == b"STOP_":
                return "STOP_", None, None
            if buffer == b"KILL_":
                self.message_end = True
                return "STOP_", None, None
            if buffer == b"KILL_CONFIRM":
                self.kill_progress = True
                return "KILL_", None, None
            # try:
            #     buffer = load_api_data(buffer)
            # except Exception as e:
            #     self.logger.warning("there is something error. please check it. %s", e)
            if isinstance(self.socket_manager, DebugServer):
                data = buffer
            else:
                buffer = load_api_data(buffer)
            if isinstance(buffer, bytes):
                return None, None, None
            if isinstance(buffer, str):
                return buffer, None, None

        return buffer, seq_num, proto_id

    def get_resp_wait(self, seq_num, timeout=100):
        return self.socket_manager.get_resp_wait(seq_num, timeout=timeout)

    def upload(self, buffer):
        if isinstance(buffer, ApiData):
            buffer = move2target_device(buffer, torch.device('cpu'))
            file_path = os.path.join(self.session_config.nfs_path, buffer.name + ".pt")
        else:
            file_path = os.path.join(self.session_config.nfs_path, buffer + f"_{int(time.time())}")

        try:
            save_pkl(buffer, file_path)
        except Exception as e:
            self.logger.warning("there is something error in save_pt. please check it. %s", e)

    def download(self):
        buffer = None
        cur_file = None
        for file_type in ("start*", "*.pt", "end*"):
            pattern = os.path.join(self.nfs_path, file_type)
            files = glob.glob(pattern)
            if len(files) > 0:
                cur_file = files[0]
                break

        if cur_file is not None:
            try:
                buffer = load_pkl(cur_file)
            except Exception as e:
                self.logger.warning("there is something error. please check it. %s", e)
            remove_path(cur_file)
        return buffer

    def config_as_debug(self, func):
        if hasattr(self.socket_manager, 'set_debug_func'):
            self.socket_manager.set_debug_func(func)


class Const:
    SEP = "."
    DISTRIBUTED = 'Distributed'


class GPUService:
    def __init__(self, config):
        self.first_start = True
        self.config = config
        self.attl = None
        self.current_rank = 0

    def exec(self, name, current_iter, current_rank, args, kwargs):
        self.current_rank = current_rank
        if isinstance(self.attl.socket_manager, DebugServer):
            self.attl.socket_manager.set_device(args[0].device)
        api_data = ApiData(
            name=name,
            args=args,
            kwargs=kwargs,
            result=None,
            step=current_iter,
            rank=current_rank,
            time=time.time()
        )
        seq_num = self.attl_send(api_data)    #发送数据时，接受 sequence_number
        logger.debug(f"gpu-fallback client send seq_num:{seq_num} step:{current_iter} rank:{current_rank} op_name: {name}")
        if seq_num is not None:
            # server data UtDataInfo
            result = self.attl.get_resp_wait(seq_num, timeout=1000)     #通过 sequence_number 获取服务端响应数据
            if isinstance(self.attl.socket_manager, DebugServer):
                api_data = result
            else:
                api_data = load_api_data(result)
            if not isinstance(api_data, ApiData):
                api_data = ApiData(*api_data)

            logger.debug(f"gpu-fallback client receive seq_num:{seq_num} step:{current_iter} rank:{current_rank} op_name: {name} data type:{type(api_data)}")
            target_device = 'device' #args[0].device
            if hasattr(api_data.result, 'in_fwd_data_list'):
                args_bench, kwargs_bench = move2target_device_(api_data.result.in_fwd_data_list[0],target_device), api_data.result.in_fwd_data_list[1]
            else:
                args_bench, kwargs_bench = move2target_device_(api_data.args,target_device), api_data.kwargs
            if hasattr(api_data.result, 'bench_output'):
                if api_data.result.bench_output is not None:
                    if isinstance(api_data.result.bench_output, tuple):
                        fallback_result = [x.to(target_device) for x in api_data.result.bench_output]
                    else:
                        fallback_result = api_data.result.bench_output.to(target_device)
            else:
                fallback_result = move2target_device_(api_data.result, target_device)
            # assert len(args) == len(args_bench), \
            #     f"Length mismatch: len(args) = {len(args)}, len(args_bench) = {len(args_bench)}"
            # 对args进行更新
            # for i in range(min(len(args), len(args_bench))):
            #     if isinstance(args[i], torch.Tensor) and isinstance(args_bench[i], torch.Tensor):
            #         args[i].data = args_bench[i]
            # # 对kwargs进行更新
            # kwargs.update(kwargs_bench)
        fallback_flag = True
        return args, kwargs, fallback_result, fallback_flag

    def attl_send(self, api_data):
        logger.info(f"tools is dumping api: {api_data.name}, rank: {self.current_rank}")
        # api_type, _, _ = api_data.name.split(Const.SEP)
        api_type = api_data.name
        if api_type in [Const.DISTRIBUTED]:
            logger.info(f"api {api_data.name} is not supported, skip")
            return
        if self.config.nfs_path:
            self.attl.upload(api_data)
            return None
        else:
            return self.attl.send(api_data)

    def start(self):
        if self.first_start:
            self.attl_init()
            self.first_start = False

    def stop(self):
        if self.attl:
            self.attl_stop()

    def attl_init(self):
        if self.config.is_online:
            attl_config = ATTLConfig(is_benchmark_device=False,
                                     connect_ip=self.config.host,
                                     connect_port=self.config.port,
                                     nfs_path=self.config.nfs_path,
                                     tls_path=self.config.tls_path)
            need_dump = len(self.config.rank) == 0 or self.current_rank in self.config.rank
            self.attl = ATTL('device', attl_config, need_dump=need_dump)
            if self.config.nfs_path:
                self.attl.upload("start")

    def attl_stop(self):
        if self.config.nfs_path:
            self.attl.upload("end")
        elif self.attl.socket_manager is not None:
            logger.info(f"pid: {os.getpid()} finished, start send STOP signal.")
            self.attl.socket_manager.send_stop_signal()

    def config_as_debug(self, func):
        self.attl.config_as_debug(func)
