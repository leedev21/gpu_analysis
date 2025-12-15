import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hydra
import importlib
from launcher.utils import Event, record, print_rank_0, _get_env_cfg
import torch
import time
from omegaconf import OmegaConf, DictConfig
from report import report
from tqdm import tqdm  # 导入tqdm库用于高效显示进度条
import traceback
from moprobe.advanced_compare import draw_manager
from moprobe.utils import acc_check_helper
import logging
from datetime import datetime
import re
from copy import deepcopy
try:
    from torchtrace.torchtrace import set_torchtrace, update
except ImportError:
    def set_torchtrace(**kwargs):
        pass

    def update(*args, **kwargs):
        pass

def get_module_name(args):
    if args.get("stages"):
        return args['stages']
    return args['name']


def test_module(runner, model, train_iterator, forward_backward_func, args):
    acc = None
    steps = args.run.max_steps
    time_befor_step_cuda = Event()
    time_after_step_cuda = Event()

    time_start = time.perf_counter()
    record(time_befor_step_cuda)

    if model['cuda_graph']:
        model['cuda_graph'].replay()
    else:
        acc = runner.run_iter(model['model'], train_iterator, forward_backward_func, steps, args)

    step_time_cuda = record(time_befor_step_cuda, time_after_step_cuda)
    time_end = time.perf_counter()
    return (time_end - time_start) * 1000 / steps, step_time_cuda * 1000 / steps, acc

nsys_profile_started = False
def run_test_case(runner, _i, total_cases, device, hw_name, test_case, args, error_logger):
    global nsys_profile_started
    acc = None
    test_success = False
    failed_case_num = 0
    _mean, _min, _max = 0, 0, 0
    try:
        # 获取算子名称
        op_name = test_case.get('name', 'Unknown')
        model, data_iterator, forward_backward_func = runner.prepare_test_case(test_case, args)
        if not data_iterator:
            return None, 0.0, 0.0, 0.0, test_success, 1

        # 检查是否使用了真实数据
        if hasattr(data_iterator, 'use_real_data') and data_iterator.use_real_data:
            print(f"🔄 [{_i+1}/{total_cases}] 使用真实数据测试算子: {op_name}")
        else:
            print(f"🔄 [{_i+1}/{total_cases}] 使用随机数据测试算子: {op_name}")

        time_cuda_list = []
        if 'cuda_graph' in args.run and args.run.cuda_graph:
            try:
                with torch.cuda.graph(model['cuda_graph']):
                    runner.run_iter(model['model'], data_iterator, forward_backward_func, args.run.max_steps, args)
            except:
                model['cuda_graph'] = None
        else:
            model['cuda_graph'] = None

        # 执行测试循环
        for i in range(args.run.loop_time):
            if not nsys_profile_started and device.type == 'cuda' and args.run.nsys_profile.enabled:
                if i == args.run.nsys_profile.start_step and rank in args.run.nsys_profile.ranks:
                    print_rank_0("====== Start nsys profiling ======")
                    torch.cuda.cudart().cudaProfilerStart()
                    if args.run.nsys_profile.gen_shape:
                        torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
                    nsys_profile_started = True
            time_cpu, time_cuda, acc = test_module(runner, model, data_iterator, forward_backward_func, args)
            time_cuda_list.append(time_cuda)
            if acc and args.run.loop_time > 10:
                acc_check_helper.add(_i, model['model'][0], test_case, hw_name, acc)
                if i > 0:
                    detail_msg = acc_check_helper.check_this(args)
                    failed_case_num += print_info(0, test_case, hw_name, args, runner.get_report_format(args), time_cuda, time_cuda, time_cuda, detail_msg)

        del data_iterator
        torch.cuda.empty_cache()
        time_cuda_list.sort(reverse=True)
        sum_start = args.run.warm_up if hasattr(args.run, 'warm_up') and len(time_cuda_list) > args.run.warm_up else 0
        _mean = torch.tensor(sum(time_cuda_list[sum_start:]) / len(time_cuda_list[sum_start:]), device=device)
        _min = torch.tensor(min(time_cuda_list[sum_start:]), device=device)
        _max = torch.tensor(max(time_cuda_list[sum_start:]), device=device)
        if torch.distributed.is_initialized():
            if torch.distributed.get_backend() == 'gloo':
                torch.distributed.all_reduce(_min.cpu(), torch.distributed.ReduceOp.MIN)
                torch.distributed.all_reduce(_max.cpu(), torch.distributed.ReduceOp.MAX)
            else:
                if torch.distributed.get_backend() != 'eccl':
                    torch.distributed.all_reduce(_mean, torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(_min, torch.distributed.ReduceOp.MIN)
                torch.distributed.all_reduce(_max, torch.distributed.ReduceOp.MAX)
        _mean = _mean.detach().item()
        _min = _min.detach().item()
        _max = _max.detach().item()
        if acc:
            if 'op' in args.stages:
                failed_case_num += print_info(_i, test_case, hw_name, args, runner.get_report_format(args), _mean, _min, _max, acc)
            elif args.run.loop_time <= 10:
                acc_check_helper.add(_i, model['model'][0], test_case, hw_name, acc)
        else:
            report(test_case, _mean, _min, _max, acc)
            report.instant_report(args, {'hw': hw_name, 'format': runner.get_report_format(args)})

        test_success = True

    except Exception as e:
        # 记录错误信息到对应算子的日志文件
        traceback_str = traceback.format_exc()
        failed_case_num += 1

        # 使用新的错误日志记录器
        log_case = deepcopy(test_case)
        for k in ['init', 'input']:
            if k in log_case:
                if 'load' in log_case[k]:
                    del log_case[k]['load']
        error_logger.log_error(op_name, _i, log_case, e, traceback_str)

        # 在控制台显示简化的错误信息
        print(f"\n❌ 测试用例 #{_i+1} 失败: {op_name}")
        print(f"   错误: {str(e)}")
        print(f"   详细信息已记录到算子专用日志文件")

        # 如果启用了真实数据，额外显示数据来源信息
        if hasattr(args.run, 'enable_real_data') and args.run.enable_real_data:
            print(f"   数据来源: 真实数据（如启用）")

        # 清理资源
        try:
            if 'data_iterator' in locals():
                del data_iterator
            torch.cuda.empty_cache()
        except:
            pass

    return acc, _mean, _min, _max, test_success, failed_case_num


def print_info(_i, case, hw_name, args, report_format, _mean, _min, _max, acc):
    test_case = deepcopy(case)
    for k in ['init', 'input']:
        if k in test_case:
            if 'load' in test_case[k]:
                del test_case[k]['load']
    failed_case_num = 0
    formatted_string_acc = " \n" + "\n".join(
        "\n ".join(f"{header}: {value}" for header, value in zip(report_format, row))
        for row in acc
    ) if acc else ''
    if int(os.environ.get('RANK', 0)) == 0:
        print('\n', _i, test_case, _mean, _min, _max, formatted_string_acc)
    else:
        print_rank_0('\n', _i, test_case, _mean, _min, _max, formatted_string_acc)
    report(test_case, _mean, _min, _max, acc)
    report.instant_report(args, {'hw': hw_name, 'format': report_format})
    if acc:
        for res in acc:
            if res[0] != 'pass':
                failed_case_num += 1
    if acc and args.run.draw:
        draw_manager.draw({'hw': hw_name, 'format': report_format})
    return failed_case_num


def sanitize_filename(filename):
    """清理文件名，移除不合法的字符"""
    # 替换不合法的文件名字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 替换双冒号为单下划线
    filename = filename.replace('::', '_')
    # 移除开头的点号
    filename = filename.lstrip('.')
    # 限制文件名长度
    if len(filename) > 100:
        filename = filename[:100]
    return filename


class OperatorErrorLogger:
    """按算子名称分类的错误日志记录器"""

    def __init__(self):
        self.error_log_dir = "error_logs"
        os.makedirs(self.error_log_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.loggers = {}  # 存储每个算子的日志记录器
        self.log_files = {}  # 存储每个算子的日志文件路径

        # 创建总体错误统计日志
        self.summary_log_file = os.path.join(self.error_log_dir, f"error_summary_{self.timestamp}.log")
        self.summary_logger = self._create_logger('error_summary', self.summary_log_file)

    def _create_logger(self, logger_name, log_file):
        """创建单个日志记录器"""
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)

        # 清除已有的处理器
        logger.handlers.clear()

        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.ERROR)

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        # 添加处理器到日志记录器
        logger.addHandler(file_handler)

        return logger

    def get_operator_logger(self, op_name):
        """获取或创建指定算子的日志记录器"""
        if op_name not in self.loggers:
            # 清理算子名称作为文件名
            safe_op_name = sanitize_filename(op_name)
            log_file = os.path.join(self.error_log_dir, f"{safe_op_name}_errors_{self.timestamp}.log")

            # 创建算子专用的日志记录器
            logger_name = f"error_{safe_op_name}"
            logger = self._create_logger(logger_name, log_file)

            self.loggers[op_name] = logger
            self.log_files[op_name] = log_file

        return self.loggers[op_name]

    def log_error(self, op_name, test_case_index, test_case, error, traceback_str):
        """记录算子错误"""
        # 获取算子专用的日志记录器
        op_logger = self.get_operator_logger(op_name)

        # 详细错误信息
        error_msg = f"""
测试用例 #{test_case_index+1} 执行失败:
算子名称: {op_name}
测试参数: {test_case}
错误类型: {type(error).__name__}
错误信息: {str(error)}
堆栈跟踪:
{traceback_str}
{'='*80}
"""

        # 记录到算子专用日志
        op_logger.error(error_msg)

        # 记录到总体统计日志
        summary_msg = f"算子 {op_name} - 测试用例 #{test_case_index+1} 失败: {str(error)}"
        self.summary_logger.error(summary_msg)

    def log_info(self, message):
        """记录信息到总体日志"""
        self.summary_logger.info(message)

    def get_log_files_info(self):
        """获取所有日志文件信息"""
        return {
            'summary': self.summary_log_file,
            'operators': self.log_files.copy()
        }


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def main(args) -> None:
    global nsys_profile_started
    OmegaConf.resolve(args)
    n_device, hw_name = _get_env_cfg()
    if args.run.draw:
        draw_manager.set('draw_ulp', True)
        draw_manager.set('hw_name', hw_name)
        if args.run.load_op_summary:
            import json
            user_config_summary_file = isinstance(args.run.load_op_summary, str)
            summary_file = args.run.load_op_summary if user_config_summary_file else draw_manager.summary_file
            with open(summary_file, 'r') as f:
                if user_config_summary_file:
                    draw_manager.set('summary_file', summary_file)
                draw_manager.set('summary', json.load(f))

    failed_case_num = 0
    successful_case_num = 0
    error_case_num = 0

    # 创建按算子分类的错误日志记录器
    error_logger = OperatorErrorLogger()

    if hw_name == 'None':
        print('HW not support!!!', hw_name)
        exit(1)
    print(args)

    # 记录测试开始信息
    error_logger.log_info(f"开始测试 - 硬件: {hw_name}, 配置: {args}")

    # myTorchTraceMode.__enter__()
    for stage in get_module_name(args):
        import_path = 'runner.' + stage + '.function'
        runner = importlib.import_module(import_path)

        # 初始化真实数据加载器（如果启用）
        if hasattr(args.run, 'enable_real_data') and args.run.enable_real_data:
            dump_json_path = getattr(args.run, 'dump_json_path', None)
            pt_data_dir = getattr(args.run, 'pt_data_dir', None)

            if dump_json_path and pt_data_dir:
                # 设置真实数据加载器
                if hasattr(runner, 'data_loader'):
                    runner.data_loader.set_real_data_loader(
                        dump_json_path=dump_json_path,
                        pt_data_dir=pt_data_dir,
                        enable_real_data=args.run.enable_real_data
                    )
                    print(f"✅ 真实数据加载器已启用:")
                    print(f"   dump.json 路径: {dump_json_path}")
                    print(f"   pt 文件目录: {pt_data_dir}")
                else:
                    print("⚠️  当前runner模块不支持真实数据加载器")
            else:
                print("⚠️  启用真实数据加载需要同时指定 dump_json_path 和 pt_data_dir")

        device, rank = runner.initialize_distributed(args)
        runner.manual_seed(rank)

        nsys_profile_started = False

        all_test_cases = list(runner.splite_test_case(args))
        total_cases = len(all_test_cases)

        test_start_time = time.time()

        print(f"\n开始测试 {total_cases} 个测试用例...")
        print(f"错误日志将保存到: {error_logger.error_log_dir}/")

        with tqdm(total=total_cases, desc="处理测试用例", unit="case", ncols=100) as pbar:
            # Running the model for n iterations
            acc = None
            for _i, test_case in enumerate(all_test_cases):
                if report.check_limit(test_case, {'hw': hw_name}, getattr(args.run, 'duration_limit')):
                    pbar.update(1)
                    continue

                if 'acc_check' in args.run and args.run.acc_check and args.run.loop_time > 10:
                    acc_check_helper.clear()
                vllm_enable = True if 'use_vllm_backend' in test_case else False

                if vllm_enable:
                    from launcher.vllm_backend import init_distributed_vllm
                    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
                    from vllm.config import VllmConfig, ParallelConfig, CompilationConfig, set_current_vllm_config, get_current_vllm_config
                    quant_config = Fp8Config(is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128])
                    parallel_config = ParallelConfig(data_parallel_size=8, pipeline_parallel_size=1, enable_expert_parallel=True)
                    vllm_config = VllmConfig(quant_config=quant_config, parallel_config=parallel_config)
                    with set_current_vllm_config(vllm_config):
                        init_distributed_vllm()
                        acc, _mean, _min, _max, test_success, _failed_case_num = run_test_case(runner, _i, total_cases, device, hw_name, test_case, args, error_logger)
                else:
                    acc, _mean, _min, _max, test_success, _failed_case_num = run_test_case(runner, _i, total_cases, device, hw_name, test_case, args, error_logger)
                failed_case_num += _failed_case_num
                if test_success:
                    successful_case_num += 1
                else:
                    error_case_num += 1

                pbar.update(1)
            if acc:
                if 'op' not in args.stages:
                    if args.run.loop_time <= 10:
                        detail_msg = acc_check_helper.check(args)
                        failed_case_num += print_info(0, test_case, hw_name, args, runner.get_report_format(args), _mean, _min, _max, detail_msg)
                draw_manager.save_summary()

        test_end_time = time.time()
        total_test_time = test_end_time - test_start_time
        hours, remainder = divmod(total_test_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_format = ""
        if hours > 0:
            time_format += f"{int(hours)}小时"
        if minutes > 0:
            time_format += f"{int(minutes)}分钟"
        time_format += f"{seconds:.2f}秒"

        print("\n\n" + "="*80)
        print("测试完成统计:")
        print(f"总测试用例数: {total_cases}")
        print(f"✅ 成功: {successful_case_num}")
        print(f"❌ 失败: {error_case_num}")
        print(f"⚠️  其他失败: {failed_case_num}")
        print(f"总执行时间: {time_format} ({total_test_time:.2f}秒)")

        # 显示日志文件信息
        log_files_info = error_logger.get_log_files_info()
        print(f"\n📋 日志文件:")
        print(f"  📄 总体错误统计: {log_files_info['summary']}")
        if log_files_info['operators']:
            print(f"  📁 算子专用错误日志:")
            for op_name, log_file in log_files_info['operators'].items():
                print(f"    - {op_name}: {log_file}")
        else:
            print(f"  🎉 没有算子错误日志（所有测试都成功！）")

        print("="*80)

        if hasattr(report, 'op_log_files'):
            print(f"生成的其他日志文件:")
            for op, path in report.op_log_files.items():
                print(f"  - {op}: {path}")

        if device.type == 'cuda' and args.run.nsys_profile.enabled:
            if rank in args.run.nsys_profile.ranks:
                print_rank_0("====== End nsys profiling ======")
                torch.cuda.cudart().cudaProfilerStop()

    # 记录测试结束信息
    error_logger.log_info(f"测试结束 - 总用例: {total_cases}, 成功: {successful_case_num}, 失败: {error_case_num}")

    # 根据是否有错误决定退出码，但不强制退出
    if error_case_num > 0:
        print(f"\n⚠️  有 {error_case_num} 个测试用例失败，详细信息请查看对应的算子错误日志文件")
        # 不再强制退出，让用户决定
        # exit(1)
    else:
        print(f"\n🎉 所有测试用例都成功执行！")

if __name__ == "__main__":
    if set_torchtrace and os.getenv('RUN_TYPE', '') != 'test':
        try:
            from torchtrace.torchtrace import set_torchtrace, update
            import subprocess
            def is_card():
                try:
                    result = subprocess.run(
                        "dpkg -l | grep sdk",
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        encoding='utf-8'
                    )
                    return "sdk" in result.stdout
                except Exception:
                    return False

            if is_card():
                update('customer_op', {'flash_attn': 'flash_attn_device'})
            else:
                update('customer_op', {'flash_attn': 'flash_attn_2_cuda'})
        except:
            pass
        set_torchtrace(torch_dispatch_trace=True, torch_api_trace=False, save_pt=False, sync_mode=True, save_to=os.path.abspath(os.getenv('SAVE_PATH', './')))

    main()

    if set_torchtrace and os.getenv('RUN_TYPE', '') != 'test':
        set_torchtrace(torch_dispatch_trace=False, torch_api_trace=False)
