# Copyright (c) 2024-2024, Huawei Technologies Co., Ltd.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0  (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import numpy as np

from .compare_utils import Const, CompareConst


def format_value(value):
    return float('{:.12f}'.format(value))

def handle_inf_nan(n_value, b_value):
    """处理inf和nan的数据"""
    n_inf = np.isinf(n_value)
    b_inf = np.isinf(b_value)
    n_nan = np.isnan(n_value)
    b_nan = np.isnan(b_value)
    n_invalid = np.any(n_inf) or np.any(n_nan)
    b_invalid = np.any(b_inf) or np.any(b_nan)
    if n_invalid or b_invalid:
        if np.array_equal(n_inf, b_inf) and np.array_equal(n_nan, b_nan):
            n_value[n_inf] = 0
            b_value[b_inf] = 0
            n_value[n_nan] = 0
            b_value[b_nan] = 0
        else:
            return CompareConst.NAN, CompareConst.NAN
    return n_value, b_value


def get_error_type(n_value, b_value, error_flag):
    """判断数据是否有异常并返回异常的n_value, b_value，同时返回error_flag"""
    if error_flag:
        return CompareConst.READ_NONE, CompareConst.READ_NONE, True
    if n_value.size == 0:  # 判断读取到的数据是否为空
        return CompareConst.NONE, CompareConst.NONE, True
    if n_value.shape != b_value.shape:  # 判断NPU和bench的数据结构是否一致
        return CompareConst.SHAPE_UNMATCH, CompareConst.SHAPE_UNMATCH, True
    if not n_value.shape:  # 判断数据是否为标量
        return n_value, b_value, False

    n_value, b_value = handle_inf_nan(n_value, b_value)  # 判断是否有nan/inf数据
    if n_value is CompareConst.NAN or b_value is CompareConst.NAN:
        return CompareConst.NAN, CompareConst.NAN, True
    return n_value, b_value, False


def reshape_value(n_value, b_value):
    """返回reshape后的数据"""
    if not n_value.shape:  # 判断数据是否为标量
        if n_value.dtype == bool:
            n_value = n_value.astype(float)
            b_value = b_value.astype(float)
        return n_value, b_value

    n_value = n_value.reshape(-1).astype(float)
    b_value = b_value.reshape(-1).astype(float)
    return n_value, b_value


def get_error_message(n_value, b_value, npu_op_name, error_flag, error_file=None):
    """获取异常情况的错误信息"""
    if error_flag:
        if n_value == CompareConst.READ_NONE:
            if error_file:
                return "Dump file: {} not found.".format(error_file)
            return CompareConst.NO_BENCH
        if n_value == CompareConst.NONE:
            return "This is empty data, can not compare."
        if n_value == CompareConst.SHAPE_UNMATCH:
            return "Shape of device and bench Tensor do not match. Skipped."
        if n_value == CompareConst.NAN:
            return "The position of inf or nan in device and bench Tensor do not match."
    else:
        if not n_value.shape:
            return "This is type of scalar data, can not compare."
        if n_value.dtype != b_value.dtype:
            print("Dtype of device and bench Tensor do not match: {}".format(npu_op_name))
            return "Dtype of device and bench Tensor do not match."
    return ""


def npy_data_check(n_value, b_value):
    error_message = ""
    if not isinstance(n_value, np.ndarray) or not isinstance(b_value, np.ndarray):
        error_message += "Dump file is not ndarray.\n"

    # 检查 n_value 和 b_value 是否为空
    if not error_message and (n_value.size == 0 or b_value.size == 0):
        error_message += "This is empty data, can not compare.\n"

    if not error_message:
        if not n_value.shape or not b_value.shape:
            error_message += "This is type of scalar data, can not compare.\n"
        if n_value.shape != b_value.shape:
            error_message += "Shape of device and bench Tensor do not match.\n"
        if n_value.dtype != b_value.dtype:
            error_message += "Dtype of device and bench Tensor do not match. Skipped.\n"

    if not error_message:
        n_value, b_value = handle_inf_nan(n_value, b_value)  # 判断是否有 nan/inf 数据
        # handle_inf_nan 会返回'Nan'或ndarray类型，使用类型判断是否存在无法处理的nan/inf数据
        if not isinstance(n_value, np.ndarray) or not isinstance(b_value, np.ndarray):
            error_message += "The position of inf or nan in device and bench Tensor do not match.\n"
    if error_message == "":
        error_flag = False
    else:
        error_flag = True
    return error_flag, error_message


def statistics_data_check(result_dict):
    error_message = ""

    if result_dict.get(CompareConst.NPU_NAME) is None or result_dict.get(CompareConst.BENCH_NAME) is None:
        error_message += "Dump file not found.\n"

    if not result_dict.get(CompareConst.NPU_SHAPE) or not result_dict.get(CompareConst.BENCH_SHAPE):
        error_message += "This is type of scalar data, can not compare.\n"
    elif result_dict.get(CompareConst.NPU_SHAPE) != result_dict.get(CompareConst.BENCH_SHAPE):
        error_message += "Tensor shapes do not match.\n"

    if result_dict.get(CompareConst.NPU_DTYPE) != result_dict.get(CompareConst.BENCH_DTYPE):
        error_message += "Dtype of device and bench Tensor do not match. Skipped.\n"

    if error_message == "":
        error_flag = False
    else:
        error_flag = True
    return error_flag, error_message


class TensorComparisonBasic(abc.ABC):
    """NPU和bench中npy数据的比较模板"""
    @abc.abstractmethod
    def apply(self, n_value, b_value, error_flag, relative_err=None):
        raise NotImplementedError


class GetCosineSimilarity(TensorComparisonBasic):
    """计算cosine相似度"""
    @staticmethod
    def correct_data(result):
        if result == CompareConst.NAN:
            return result
        if float(result) > CompareConst.COSINE_THRESHOLD:
            return round(float(result), 6)
        return result

    def apply(self, n_value, b_value, error_flag, relative_err=None):
        if error_flag:
            if n_value == CompareConst.READ_NONE:
                return CompareConst.NONE, ''
            if n_value == CompareConst.NONE:
                return CompareConst.UNSUPPORTED, ''
            if n_value == CompareConst.SHAPE_UNMATCH:
                return CompareConst.SHAPE_UNMATCH, ''
            if n_value == CompareConst.NAN:
                return CompareConst.N_A, ''

        if not n_value.shape:
            return CompareConst.UNSUPPORTED, ''

        with np.errstate(divide='ignore', invalid='ignore'):
            if len(n_value) == 1:
                return CompareConst.UNSUPPORTED, "This tensor is scalar."
            num = n_value.dot(b_value)
            a_norm = np.linalg.norm(n_value)
            b_norm = np.linalg.norm(b_value)

            if a_norm <= Const.FLOAT_EPSILON and b_norm <= Const.FLOAT_EPSILON:
                return 1.0, ''
            if a_norm <= Const.FLOAT_EPSILON:
                return CompareConst.NAN, 'Cannot compare by Cosine Similarity, All the data is Zero in device dump data.'
            if b_norm <= Const.FLOAT_EPSILON:
                return CompareConst.NAN, 'Cannot compare by Cosine Similarity, All the data is Zero in Bench dump data.'

            cos = num / (a_norm * b_norm)
            if np.isnan(cos):
                return CompareConst.NAN, 'Cannot compare by Cosine Similarity, the dump data has NaN.'
            result = format_value(cos)
            result = self.correct_data(result)
        return 1.0 if float(result) > 0.99999 else result, ''


class GetMaxAbsErr(TensorComparisonBasic):
    """计算最大绝对误差"""
    def apply(self, n_value, b_value, error_flag, relative_err=None):
        if error_flag:
            if n_value == CompareConst.READ_NONE:
                return CompareConst.NONE, ""
            if n_value == CompareConst.NONE:
                return 0, ""
            if n_value == CompareConst.SHAPE_UNMATCH:
                return CompareConst.SHAPE_UNMATCH, ""
            if n_value == CompareConst.NAN:
                return CompareConst.N_A, ""

        temp_res = n_value - b_value
        max_value = np.max(np.abs(temp_res))
        if np.isnan(max_value):
            message = 'Cannot compare by MaxRelativeError, the data contains nan/inf/-inf in dump data.'
            return CompareConst.NAN, message
        return format_value(max_value), ""


def get_relative_err(n_value, b_value):
    """计算相对误差"""
    with np.errstate(divide='ignore', invalid='ignore'):
        if b_value.dtype not in CompareConst.FLOAT_TYPE:
            n_value, b_value = n_value.astype(float), b_value.astype(float)
        zero_mask = (b_value == 0)
        b_value[zero_mask] += np.finfo(b_value.dtype).eps
        n_value[zero_mask] += np.finfo(b_value.dtype).eps
        relative_err = np.divide((n_value - b_value), b_value)
    return np.abs(relative_err)


class GetMaxRelativeErr(TensorComparisonBasic):
    """计算最大相对误差"""
    def apply(self, n_value, b_value, error_flag, relative_err=None):
        if error_flag:
            if n_value == CompareConst.READ_NONE:
                return CompareConst.NONE, ''
            if n_value == CompareConst.NONE:
                return 0, ''
            if n_value == CompareConst.SHAPE_UNMATCH:
                return CompareConst.SHAPE_UNMATCH, ''
            if n_value == CompareConst.NAN:
                return CompareConst.N_A, ''

        if relative_err is None:
            relative_err = get_relative_err(n_value, b_value)
        max_relative_err = np.max(np.abs(relative_err))
        if np.isnan(max_relative_err):
            message = 'Cannot compare by MaxRelativeError, the data contains nan/inf/-inf in dump data.'
            return CompareConst.NAN, message
        return format_value(max_relative_err), ''


class GetThousandErrRatio(TensorComparisonBasic):
    """计算相对误差小于千分之一的比例"""
    def apply(self, n_value, b_value, error_flag, relative_err=None):
        if error_flag:
            if n_value == CompareConst.READ_NONE:
                return CompareConst.NONE, ""
            if n_value == CompareConst.NONE:
                return 0, ""
            if n_value == CompareConst.SHAPE_UNMATCH:
                return CompareConst.SHAPE_UNMATCH, ""
            if n_value == CompareConst.NAN:
                return CompareConst.N_A, ""

        if not n_value.shape:
            return CompareConst.NAN, ""
        if relative_err is None:
            relative_err = get_relative_err(n_value, b_value)
        if not np.size(relative_err):
            return CompareConst.NAN, ""
        return format_value(np.sum(relative_err < CompareConst.THOUSAND_RATIO_THRESHOLD) / np.size(relative_err)), ""


class GetFiveThousandErrRatio(TensorComparisonBasic):
    """计算相对误差小于千分之五的比例"""
    def apply(self, n_value, b_value, error_flag, relative_err=None):
        if error_flag:
            if n_value == CompareConst.READ_NONE:
                return CompareConst.NONE, ""
            if n_value == CompareConst.NONE:
                return 0, ""
            if n_value == CompareConst.SHAPE_UNMATCH:
                return CompareConst.SHAPE_UNMATCH, ""
            if n_value == CompareConst.NAN:
                return CompareConst.N_A, ""

        if not n_value.shape:
            return CompareConst.NAN, ""
        if relative_err is None:
            relative_err = get_relative_err(n_value, b_value)
        if not np.size(relative_err):
            return CompareConst.NAN, ""
        return format_value(
            np.sum(relative_err < CompareConst.FIVE_THOUSAND_RATIO_THRESHOLD) / np.size(relative_err)), ""


class GetHundredErrRatio(TensorComparisonBasic):
    """计算相对误差小于百分之一的比例"""
    def apply(self, n_value, b_value, error_flag, relative_err=None):
        if error_flag:
            if n_value == CompareConst.READ_NONE:
                return CompareConst.NONE, ""
            if n_value == CompareConst.NONE:
                return 0, ""
            if n_value == CompareConst.SHAPE_UNMATCH:
                return CompareConst.SHAPE_UNMATCH, ""
            if n_value == CompareConst.NAN:
                return CompareConst.N_A, ""

        if not n_value.shape:
            return CompareConst.NAN, ""
        if relative_err is None:
            relative_err = get_relative_err(n_value, b_value)
        if not np.size(relative_err):
            return CompareConst.NAN, ""
        return format_value(
            np.sum(relative_err < CompareConst.HUNDRED_RATIO_THRESHOLD) / np.size(relative_err)), ""


class GetTenThousandErrRatio(TensorComparisonBasic):
    """计算相对误差小于万分之一的比例"""
    def apply(self, n_value, b_value, error_flag, relative_err=None):
        if error_flag:
            if n_value == CompareConst.READ_NONE:
                return CompareConst.NONE, ""
            if n_value == CompareConst.NONE:
                return 0, ""
            if n_value == CompareConst.SHAPE_UNMATCH:
                return CompareConst.SHAPE_UNMATCH, ""
            if n_value == CompareConst.NAN:
                return CompareConst.N_A, ""

        if not n_value.shape:
            return CompareConst.NAN, ""
        if relative_err is None:
            relative_err = get_relative_err(n_value, b_value)
        if not np.size(relative_err):
            return CompareConst.NAN, ""
        return format_value(
            np.sum(relative_err < CompareConst.TEN_THOUSAND_RATIO_THRESHOLD) / np.size(relative_err)), ""


class CompareOps:
    compare_ops = {
        # "cosine_similarity": GetCosineSimilarity(),
        "max_abs_error": GetMaxAbsErr(),
        "max_relative_error": GetMaxRelativeErr(),
        "ten_thousand_err_ratio": GetTenThousandErrRatio(),
        "one_thousand_err_ratio": GetThousandErrRatio(),
        "five_thousand_err_ratio": GetFiveThousandErrRatio(),
        "hundred_err_ratio": GetHundredErrRatio(),
    }


def compare_ops_apply(n_value, b_value, error_flag, err_msg, relative_err=None):
    result_list = []
    for op in CompareOps.compare_ops.values():
        result, msg = op.apply(n_value, b_value, error_flag, relative_err=relative_err)
        err_msg += msg
        result_list.append(result)
    return result_list, err_msg
