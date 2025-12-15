import torch
import numpy as np
import json
from .algorithm import get_rmse, get_error_balance, get_max_rel_err, \
    get_mean_rel_err, get_rel_err, get_abs_err, get_max_abs_err, get_rel_err_ratio, cosine_sim, get_rel_err_origin, \
    get_small_value_err_ratio, get_finite_and_infinite_mask, get_small_value_mask, check_inf_nan_value, \
    check_small_value, check_norm_value, get_abs_bench_with_eps, get_ulp_err

from .compare_utils import check_dtype_comparable, CompareConst, CompareColumn, \
    DETAIL_TEST_ROWS, precision_configs, BENCHMARK_COMPARE_SUPPORT_LIST, absolute_standard_api, binary_standard_api, \
    ulp_standard_api, thousandth_standard_api, apis_threshold


class FileOpen:
    """
    The class for open file by a safe way.

    Attributes:
        file_path: The file or dictionary path to be opened.
        mode(str): The file open mode
    """
    SUPPORT_READ_MODE = ["r", "rb"]
    SUPPORT_WRITE_MODE = ["w", "wb", "a", "ab"]
    SUPPORT_READ_WRITE_MODE = ["r+", "rb+", "w+", "wb+", "a+", "ab+"]

    def __init__(self, file_path, mode, encoding='utf-8'):
        self.file_path = file_path
        self.mode = mode
        self.encoding = encoding
        self._handle = None

    def __enter__(self):
        binary_mode = "b"
        if binary_mode not in self.mode:
            self._handle = open(self.file_path, self.mode, encoding=self.encoding)
        else:
            self._handle = open(self.file_path, self.mode)
        return self._handle

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._handle:
            self._handle.close()

    
def get_file_content_bytes(file):
    with FileOpen(file, 'rb') as file_handle:
        return file_handle.read()

def get_json_contents(file_path):
    ops = get_file_content_bytes(file_path)
    try:
        json_obj = json.loads(ops)
    except ValueError as error:
        print('Failed to load json.')
    if not isinstance(json_obj, dict):
        print('Json file content is not a dictionary!')
    return json_obj


class Comparator:
    # consts for result csv
    COLUMN_API_NAME = "API name"
    COLUMN_FORWARD_SUCCESS = "Forward Test Success"
    COLUMN_BACKWARD_SUCCESS = "Backward Test Success"
    COLUMN_STACK_INFO = "Traceback callstack info"

    def __init__(self, result_csv_path, details_csv_path, is_continue_run_ut, stack_info_json_path=None, config=None):
        self.save_path_str = result_csv_path
        self.detail_save_path_str = details_csv_path
        self.save_path_list = [result_csv_path]
        self.detail_save_path_list = [details_csv_path]

        if config and config.online_config.is_online:
            self.save_path_str = result_csv_path.replace(".csv", "_rank{}.csv")
            self.detail_save_path_str = details_csv_path.replace(".csv", "_rank{}.csv")
            self.save_path_list = [self.save_path_str.format(rank) for rank in config.online_config.rank_list]
            self.detail_save_path_list = \
                [self.detail_save_path_str.format(rank) for rank in config.online_config.rank_list]

        # if not is_continue_run_ut:
        #     self.write_csv_title()
        if stack_info_json_path:
            self.stack_info = get_json_contents(stack_info_json_path)
        else:
            self.stack_info = None
        self.api_standard = {}

    def _compare_core_wrapper(self, api_name, bench_output, device_output, grad=False):
        if api_name not in self.api_standard:
            self.api_standard[api_name] = set()
        detailed_result_total = []
        test_final_success = CompareConst.PASS
        if isinstance(bench_output, (list, tuple)):
            status, compare_result, message = [], [], []
            if len(bench_output) > len(device_output):
                status = [CompareConst.ERROR]
                message = ["bench and device output structure is different."]
            else:
                device_output = device_output[:len(bench_output)]
                for b_out_i, n_out_i in zip(bench_output, device_output):
                    if isinstance(b_out_i, list):
                        for b_out_k, n_out_k in zip(b_out_i, n_out_i):
                            status_i, compare_result_i, message_i = self._compare_core(api_name, b_out_k, n_out_k, grad=grad)
                            status.append(status_i)
                            compare_result.append(compare_result_i)
                            message.append(message_i)
                    else:
                        status_i, compare_result_i, message_i = self._compare_core(api_name, b_out_i, n_out_i, grad=grad)
                        status.append(status_i)
                        compare_result.append(compare_result_i)
                        message.append(message_i)
        elif isinstance(bench_output, (dict)):
            status, compare_result, message = [], [], []
            if len(bench_output) > len(device_output):
                status = [CompareConst.ERROR]
                message = ["bench and device output structure is different."]
            else:
                for k in bench_output:
                    b_out_i = bench_output[k]
                    n_out_i = device_output[k]
                    if isinstance(b_out_i, list):
                        for b_out_k, n_out_k in zip(b_out_i, n_out_i):
                            status_i, compare_result_i, message_i = self._compare_core(api_name, b_out_k, n_out_k, grad=grad)
                            status.append(status_i)
                            compare_result.append(compare_result_i)
                            message.append(message_i)
                    else:
                        status_i, compare_result_i, message_i = self._compare_core(api_name, b_out_i, n_out_i, grad=grad)
                        status.append(status_i)
                        compare_result.append(compare_result_i)
                        message.append(message_i)
        else:
            status, compare_result, message = self._compare_core(api_name, bench_output, device_output, grad=grad)
        if not isinstance(status, list):
            detailed_result_total.append(compare_result.to_simple_column(status, message))
            if status == CompareConst.ERROR:
                test_final_success = CompareConst.ERROR
            elif status == CompareConst.WARNING:
                test_final_success = CompareConst.WARNING
        else:
            for item, item_status in enumerate(status):
                detailed_result_total.append(compare_result[item].to_simple_column(item_status, message[item]))
                if item_status == CompareConst.ERROR:
                    test_final_success = CompareConst.ERROR
                elif item_status == CompareConst.WARNING:
                    test_final_success = CompareConst.WARNING
        return test_final_success, detailed_result_total

    def _compare_core(self, api_name, bench_output, device_output, grad=False):
        compare_column = CompareColumn()
        if not isinstance(bench_output, type(device_output)):
            status = CompareConst.ERROR
            message = "bench and device output type is different."
        elif isinstance(bench_output, dict):
            b_keys, n_keys = set(bench_output.keys()), set(device_output.keys())
            if b_keys != n_keys:
                status = CompareConst.ERROR
                message = "bench and device output dict keys are different."
            else:
                status, compare_column, message = self._compare_core(api_name, list(bench_output.values()),
                                                                     list(device_output.values()), grad=grad)
        elif isinstance(bench_output, torch.Tensor):
            copy_bench_out = bench_output.detach().clone()
            copy_device_output = device_output.detach().clone()
            compare_column.bench_type = str(copy_bench_out.dtype)
            compare_column.npu_type = str(copy_device_output.dtype)
            compare_column.shape = tuple(device_output.shape)
            status, compare_column, message = self._compare_torch_tensor(api_name, copy_bench_out, copy_device_output,
                                                                         compare_column, grad=grad)
        elif isinstance(bench_output, (bool, int, float, str, torch.dtype)):
            compare_column.bench_type = str(type(bench_output))
            compare_column.npu_type = str(type(device_output))
            status, compare_column, message = self._compare_builtin_type(bench_output, device_output, compare_column)
        elif bench_output is None:
            status = CompareConst.SKIP
            message = "Bench output is None, skip this test."
        else:
            status = CompareConst.SKIP
            message = "Unexpected output type in compare_core: {}".format(type(bench_output))

        return status, compare_column, message

    @staticmethod
    def _compare_dropout(bench_output, device_output):
        tensor_num = bench_output.numel()
        if tensor_num >= 100:
            if abs((bench_output == 0).sum() - (device_output == 0).cpu().sum()) / tensor_num < 0.1:
                return CompareConst.PASS, 1
            else:
                return CompareConst.ERROR, 0
        else:
            return CompareConst.PASS, 1

    @staticmethod
    def _compare_builtin_type(bench_output, device_output, compare_column):
        if not isinstance(bench_output, (bool, int, float, str)):
            return CompareConst.PASS, compare_column, ""
        if bench_output != device_output:
            return CompareConst.ERROR, compare_column, ""
        compare_column.error_rate = 0
        return CompareConst.PASS, compare_column, ""

    @staticmethod
    def _compare_bool_tensor(bench_output, device_output, api_name=None):
        error_nums = (bench_output != device_output).sum()
        if bench_output.size == 0:
            return CompareConst.NAN, CompareConst.ERROR, "There is not bench calculation result."
        if error_nums != 0:
            if isinstance(device_output, (list, tuple)):
                error_nums_tmp = np.array([(bench_output[k] != device_output[k*2]) for k in range(len(device_output)//2)]).sum()
                if error_nums_tmp == 0:
                    error_nums = 0
        error_rate = float(error_nums / bench_output.size)
        if error_rate == 0:
            return error_rate,  CompareConst.PASS, "Error Rate is 0, consider as pass"
        else:
            return error_rate,  CompareConst.ERROR, "Error Rate is not 0, consider as error"


    @staticmethod
    def _get_absolute_threshold_attribute(api_name, dtype):
        small_value_threshold = apis_threshold.get(api_name).get(dtype).get('small_value')
        small_value_atol = apis_threshold.get(api_name).get(dtype).get('small_value_atol')
        rtol = apis_threshold.get(api_name).get(dtype).get('rtol')
        return small_value_threshold, small_value_atol, rtol

    def _compare_torch_tensor(self, api_name, bench_output, device_output, compare_column, grad=False):
        cpu_shape = bench_output.shape
        npu_shape = device_output.shape
        npu_dtype = device_output.dtype
        if grad:
            bench_output = bench_output.grad
            device_output = device_output.grad
        if npu_dtype in [torch.bfloat16, torch.float8_e4m3fn]:
            bench_output = bench_output.to(torch.float32)
            device_output = device_output.to(torch.float32)
        elif bench_output.dtype in [torch.bfloat16, torch.float8_e4m3fn]:
            bench_output = bench_output.to(torch.float32)
        bench_output = bench_output.cpu().numpy()
        device_output = device_output.cpu().numpy()
        if cpu_shape != npu_shape:
            return CompareConst.ERROR, compare_column, f"The shape of bench{str(cpu_shape)} " \
                                                       f"and device{str(npu_shape)} not equal."
        if not check_dtype_comparable(bench_output, device_output):
            return CompareConst.ERROR, compare_column, f"Bench out dtype is {bench_output.dtype} but " \
                                                       f"device output dtype is {device_output.dtype}, cannot compare."
        message = ""
        if bench_output.dtype in [bool, np.uint8, np.int8, np.int16, np.uint16, np.uint32, np.int32,
                                  np.int64, np.uint64]:
            err_rate, status, msg = self._compare_bool_tensor(bench_output, device_output, api_name=api_name)
            message += msg
            compare_column.error_rate = err_rate
            # if npu_dtype == torch.int64:
            #     status = 'pass'
            return status, compare_column, message
        else:
            status, compare_column, message = self._compare_common_float_tensor(api_name, bench_output, device_output,
                                                                         compare_column, npu_dtype)
            return status, compare_column, message

    def _compare_float_tensor(self, api_name, bench_output, device_output, compare_column, dtype):
        message = ""
        abs_bench, abs_bench_with_eps = get_abs_bench_with_eps(bench_output, dtype)
        abs_err = get_abs_err(bench_output, device_output)
        rel_err_orign = get_rel_err_origin(abs_err, abs_bench_with_eps)
        if api_name in thousandth_standard_api:
            self.api_standard[api_name].add('thousandth')
            thousand_res, thousand_status = get_rel_err_ratio(rel_err_orign, CompareConst.THOUSAND_RATIO_THRESHOLD)
            compare_column.rel_err_thousandth = thousand_res
        if str(dtype) in BENCHMARK_COMPARE_SUPPORT_LIST:
            both_finite_mask, inf_nan_mask = get_finite_and_infinite_mask(bench_output, device_output)
            if api_name in binary_standard_api:
                self.api_standard[api_name].add('binary')
                err_rate, _, _ = self._compare_bool_tensor(bench_output, device_output)
                compare_column.error_rate = err_rate
            elif api_name in absolute_standard_api:
                self.api_standard[api_name].add('absolute')
                small_value_threshold, small_value_atol, rtol = self._get_absolute_threshold_attribute(
                    api_name, str(dtype))
                rel_err = abs_err / abs_bench_with_eps
                small_value_mask = get_small_value_mask(abs_bench, both_finite_mask, small_value_threshold)
                normal_value_mask = np.logical_and(both_finite_mask, np.logical_not(small_value_mask))
                compare_column.inf_nan_error_ratio = check_inf_nan_value(inf_nan_mask, bench_output, device_output,
                                                                         dtype, rtol)
                compare_column.rel_err_ratio = check_norm_value(normal_value_mask, rel_err, rtol)
                compare_column.abs_err_ratio = check_small_value(abs_err, small_value_mask, small_value_atol)
            elif api_name in ulp_standard_api:
                self.api_standard[api_name].add('ulp')
                if bench_output.size == 0:
                    compare_column.max_ulp_error = 0
                    compare_column.mean_ulp_error = 0
                    compare_column.ulp_error_proportion = 0
                else:
                    ulp_err, ulp_dtype = get_ulp_err(bench_output, device_output, dtype)
                    compare_column.max_ulp_error = np.max(ulp_err)
                    compare_column.mean_ulp_error = np.mean(ulp_err)
                    if ulp_dtype == torch.float32:
                        compare_column.ulp_error_proportion = \
                        np.sum(ulp_err > CompareConst.ULP_FLOAT32_THRESHOLD) / bench_output.size
                    else:
                        compare_column.ulp_error_proportion = \
                            np.sum(ulp_err > CompareConst.ULP_FLOAT16_THRESHOLD) / bench_output.size
            else:
                dtype_config = precision_configs.get(dtype)
                small_value_mask = get_small_value_mask(abs_bench, both_finite_mask, dtype_config['small_value'][0])
                abs_err_greater_mask = np.greater(abs_err, dtype_config['small_value_atol'][0])
                compare_column.small_value_err_ratio = get_small_value_err_ratio(small_value_mask, abs_err_greater_mask)
                rel_err = get_rel_err(abs_err, abs_bench_with_eps, small_value_mask, inf_nan_mask)
                compare_column.rmse = get_rmse(abs_err, np.logical_or(inf_nan_mask, small_value_mask))
                compare_column.eb = get_error_balance(bench_output, device_output)
                if rel_err.size == 0:
                    if str(bench_output) == '[]' and str(device_output) == '[]':
                        return CompareConst.PASS, compare_column, 'result list is all empty.'
                    else:
                        return CompareConst.ERROR, compare_column, "Relative error result list is empty."
                compare_column.max_rel_error = get_max_rel_err(rel_err)
                compare_column.mean_rel_error = get_mean_rel_err(rel_err)

        self.api_standard[api_name].add('cosine_sim')
        cos_res, cos_status, msg = cosine_sim(bench_output, device_output)
        compare_column.cosine_sim = cos_res
        message += msg + "\n"
        if not cos_status:
            message += "Cosine similarity is less than 0.99, consider as error, skip other check and set to SPACE.\n"
            return CompareConst.ERROR, compare_column, message

        self.api_standard[api_name].add('max_abs')
        max_abs_res, max_abs_status = get_max_abs_err(abs_err)
        compare_column.max_abs_err = max_abs_res
        if max_abs_status:
            message += "Max abs error is less than 0.001, consider as pass, skip other check and set to SPACE.\n"
            return CompareConst.PASS, compare_column, message
        if dtype in [torch.float16, torch.bfloat16]:
            hundred_res, hundred_status = get_rel_err_ratio(rel_err_orign, CompareConst.HUNDRED_RATIO_THRESHOLD)
            compare_column.rel_err_hundredth = hundred_res
            if not hundred_status:
                message += "Relative error is greater than 0.01, consider as error, " \
                           "skip other check and set to SPACE.\n"
                return CompareConst.ERROR, compare_column, message

        self.api_standard[api_name].add('thousand')
        thousand_res, thousand_status = get_rel_err_ratio(rel_err_orign, CompareConst.THOUSAND_RATIO_THRESHOLD)
        compare_column.rel_err_thousandth = thousand_res
        if dtype in [torch.float16, torch.bfloat16]:
            if thousand_status:
                message += "Relative error is less than 0.001, consider as pass, skip other check and set to SPACE.\n"
                return CompareConst.PASS, compare_column, message
            message += "Relative error is greater than 0.001, consider as warning, skip other check and set to SPACE.\n"
            return CompareConst.WARNING, compare_column, message

        self.api_standard[api_name].add('ten_thousand')
        ten_thousand_res, ten_thousand_status = get_rel_err_ratio(
                                                rel_err_orign, CompareConst.TEN_THOUSAND_RATIO_THRESHOLD)
        compare_column.rel_err_ten_thousandth = ten_thousand_res
        if dtype in [torch.float32, torch.float64]:
            if not thousand_status:
                message += "Relative error is greater than 0.001, consider as error, " \
                           "skip other check and set to SPACE.\n"
                return CompareConst.ERROR, compare_column, message
            if not ten_thousand_status:
                message += "Relative error is greater than 0.0001, consider as warning, " \
                           "skip other check and set to SPACE.\n"
                return CompareConst.WARNING, compare_column, message
            message += "Relative error is less than 0.0001, consider as pass.\n"
        return CompareConst.PASS, compare_column, message

    def _compare_common_float_tensor(self, api_name, bench_output, device_output, compare_column, dtype):
        message = ""
        abs_bench, abs_bench_with_eps = get_abs_bench_with_eps(bench_output, dtype)
        abs_err = get_abs_err(bench_output, device_output)
        rel_err_orign = get_rel_err_origin(abs_err, abs_bench_with_eps)
        if api_name in thousandth_standard_api:
            self.api_standard[api_name].add('thousandth')
            thousand_res, thousand_status = get_rel_err_ratio(rel_err_orign, CompareConst.THOUSAND_RATIO_THRESHOLD)
            compare_column.rel_err_thousandth = thousand_res
        if True:
            both_finite_mask, inf_nan_mask = get_finite_and_infinite_mask(bench_output, device_output)
            self.api_standard[api_name].add('ulp')
            if bench_output.size == 0:
                compare_column.max_ulp_error = 0
                compare_column.mean_ulp_error = 0
                compare_column.ulp_error_proportion = 0
            else:
                ulp_err, ulp_dtype = get_ulp_err(bench_output, device_output, torch.float32 if dtype in [torch.float64] else dtype)
                compare_column.max_ulp_error = np.max(ulp_err)
                compare_column.mean_ulp_error = np.mean(ulp_err)
                if ulp_dtype == torch.float32:
                    compare_column.ulp_error_proportion = \
                    np.sum(ulp_err > CompareConst.ULP_FLOAT32_THRESHOLD) / bench_output.size
                else:
                    compare_column.ulp_error_proportion = \
                        np.sum(ulp_err > CompareConst.ULP_FLOAT16_THRESHOLD) / bench_output.size
            dtype_config = precision_configs.get(ulp_dtype)
            small_value_mask = get_small_value_mask(abs_bench, both_finite_mask, dtype_config['small_value'][0])
            abs_err_greater_mask = np.greater(abs_err, dtype_config['small_value_atol'][0])
            compare_column.small_value_err_ratio = get_small_value_err_ratio(small_value_mask, abs_err_greater_mask)
            rel_err = get_rel_err(abs_err, abs_bench_with_eps, small_value_mask, inf_nan_mask)
            compare_column.rmse = get_rmse(abs_err, np.logical_or(inf_nan_mask, small_value_mask))
            compare_column.eb = get_error_balance(bench_output, device_output)
            if rel_err.size == 0:
                if str(bench_output) == '[]' and str(device_output) == '[]':
                    return CompareConst.PASS, compare_column, 'result list is all empty.'
                else:
                    return CompareConst.ERROR, compare_column, "Relative error result list is empty."
            compare_column.max_rel_error = get_max_rel_err(rel_err)
            compare_column.mean_rel_error = get_mean_rel_err(rel_err)

        self.api_standard[api_name].add('cosine_sim')
        cos_res, cos_status, msg = cosine_sim(bench_output, device_output)
        compare_column.cosine_sim = cos_res
        # message += msg + "\n"
        if not cos_status:
            message += "Cosine similarity is less than 0.99, consider as error"
            return CompareConst.ERROR, compare_column, message

        self.api_standard[api_name].add('max_abs')
        max_abs_res, max_abs_status = get_max_abs_err(abs_err)
        compare_column.max_abs_err = max_abs_res
        if max_abs_status:
            message += "Max abs error is less than 0.001, consider as pass"
            return CompareConst.WARNING, compare_column, message
        if dtype in [torch.float16, torch.bfloat16]:
            hundred_res, hundred_status = get_rel_err_ratio(rel_err_orign, CompareConst.HUNDRED_RATIO_THRESHOLD)
            compare_column.rel_err_hundredth = hundred_res
            if not hundred_status:
                message += "Relative error is greater than 0.01, consider as error"
                return CompareConst.ERROR, compare_column, message

        self.api_standard[api_name].add('thousand')
        thousand_res, thousand_status = get_rel_err_ratio(rel_err_orign, CompareConst.THOUSAND_RATIO_THRESHOLD)
        compare_column.rel_err_thousandth = thousand_res
        if dtype in [torch.float16, torch.bfloat16]:
            if thousand_status:
                message += "Relative error is less than 0.001, consider as pass"
                return CompareConst.WARNING, compare_column, message
            message += "Relative error is greater than 0.001, consider as warning"
            return CompareConst.WARNING, compare_column, message

        self.api_standard[api_name].add('ten_thousand')
        ten_thousand_res, ten_thousand_status = get_rel_err_ratio(
                                                rel_err_orign, CompareConst.TEN_THOUSAND_RATIO_THRESHOLD)
        compare_column.rel_err_ten_thousandth = ten_thousand_res
        if dtype in [torch.float32, torch.float64]:
            if not thousand_status:
                message += "Relative error is greater than 0.001, consider as error"
                return CompareConst.WARNING, compare_column, message
            if not ten_thousand_status:
                message += "Relative error is greater than 0.0001, consider as warning"
                return CompareConst.WARNING, compare_column, message
            message += "Relative error is less than 0.0001, consider as pass"
        if compare_column.ulp_error_proportion == 0.0:
            message += "ulp error is less than 1, consider as pass"
            return CompareConst.PASS, compare_column, message
        else:
            message += "ulp error is greater than 1, consider as warning"
            return CompareConst.WARNING, compare_column, message