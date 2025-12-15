#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import time
import os
import math
import stat
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

current_time = time.strftime("%Y%m%d%H%M%S")
API_PRECISION_COMPARE_RESULT_FILE_NAME = "api_precision_compare_result_" + current_time + ".csv"
API_PRECISION_COMPARE_DETAILS_FILE_NAME = "api_precision_compare_details_" + current_time + ".csv"
BENCHMARK_COMPARE_SUPPORT_LIST = ['torch.float16', 'torch.bfloat16', 'torch.float32']
API_PRECISION_COMPARE_UNSUPPORT_LIST = ['torch.float64', 'torch.complex64', 'torch.complex128']
ULP_COMPARE_SUPPORT_LIST = ['torch.float16', 'torch.bfloat16', 'torch.float32']
BINARY_COMPARE_UNSUPPORT_LIST = BENCHMARK_COMPARE_SUPPORT_LIST + API_PRECISION_COMPARE_UNSUPPORT_LIST


cur_path = os.path.dirname(os.path.realpath(__file__))
standard_yaml_path = os.path.join(cur_path, "api_precision_standard.yaml")
config = OmegaConf.load(standard_yaml_path)
apis = OmegaConf.to_container(config, resolve=True)
absolute_standard_api = apis.get('AbsoluteThreshStandard')
binary_standard_api = apis.get('BinaryCompareStandard')
ulp_standard_api = apis.get('ULPStandard')
thousandth_standard_api = apis.get('ThousandthStandard')


threshold_yaml_path = os.path.join(cur_path, "api_precision_threshold.yaml")
config = OmegaConf.load(threshold_yaml_path)
apis_threshold = OmegaConf.to_container(config, resolve=True)


class CompareColumn:
    def __init__(self):
        self.bench_type = CompareConst.SPACE
        self.npu_type = CompareConst.SPACE
        self.shape = CompareConst.SPACE
        self.cosine_sim = CompareConst.SPACE
        self.max_abs_err = CompareConst.SPACE
        self.rel_err_hundredth = CompareConst.SPACE
        self.rel_err_thousandth = CompareConst.SPACE
        self.rel_err_ten_thousandth = CompareConst.SPACE
        self.error_rate = CompareConst.SPACE
        self.eb = CompareConst.SPACE
        self.rmse = CompareConst.SPACE
        self.small_value_err_ratio = CompareConst.SPACE
        self.max_rel_error = CompareConst.SPACE
        self.mean_rel_error = CompareConst.SPACE
        self.inf_nan_error_ratio = CompareConst.SPACE
        self.rel_err_ratio = CompareConst.SPACE
        self.abs_err_ratio = CompareConst.SPACE
        self.max_ulp_error = CompareConst.SPACE
        self.mean_ulp_error = CompareConst.SPACE
        self.ulp_error_proportion = CompareConst.SPACE

    def to_column_value(self, is_pass, message):
        return [self.bench_type, self.npu_type, self.shape, self.cosine_sim, self.max_abs_err, self.rel_err_hundredth,
                self.rel_err_thousandth, self.rel_err_ten_thousandth, self.error_rate, self.eb, self.rmse, 
                self.small_value_err_ratio, self.max_rel_error, self.mean_rel_error, self.inf_nan_error_ratio, 
                self.rel_err_ratio, self.abs_err_ratio, self.max_ulp_error, self.mean_ulp_error, 
                self.ulp_error_proportion, is_pass, message]

    def to_simple_column(self, is_pass, message):
        return [self.cosine_sim, self.max_abs_err, self.rel_err_hundredth,
                self.rel_err_thousandth, self.rel_err_ten_thousandth, self.error_rate, self.eb, self.rmse,
                self.small_value_err_ratio, self.max_rel_error, self.mean_rel_error,
                self.max_ulp_error, self.mean_ulp_error,
                self.ulp_error_proportion, message]
    

class Const:
    """
    Class for const
    """
    TOOL_NAME = "msprobe"

    SEP = "."
    REGEX_PREFIX_MAX_LENGTH = 20
    REGEX_PREFIX_PATTERN = r"^[a-zA-Z0-9_-]+$"
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    STRING_BLACKLIST = r"^[＋－＝％＠\+\-=%@]|;[＋－＝％＠\+\-=%@]"
    COMMA = ","
    FLOAT_EPSILON = np.finfo(float).eps
    OFF = 'OFF'
    BACKWARD = 'backward'
    FORWARD = 'forward'
    JIT = 'Jit'
    PRIMITIVE_PREFIX = 'Primitive'
    DEFAULT_LIST = []
    DEFAULT_PATH = './'
    WHITE_LIST = 'white_list'
    BLACK_LIST = 'black_list'
    DUMP_TENSOR_DATA = 'dump_tensor_data'
    NONE = None
    THREE_SEGMENT = 3
    FOUR_SEGMENT = 4
    SIX_SEGMENT = 6
    SEVEN_SEGMENT = 7
    MAX_DEPTH = 10

    # dump mode
    ALL = "all"
    LIST = "list"
    RANGE = "range"
    STACK = "stack"
    ACL = "acl"
    API_LIST = "api_list"
    API_STACK = "api_stack"
    DUMP_MODE = [ALL, LIST, RANGE, STACK, ACL, API_LIST, API_STACK]
    AUTO = "auto"
    ONLINE_DUMP_MODE = [ALL, LIST, AUTO, OFF]
    SUMMARY = "summary"
    MD5 = "md5"
    SUMMARY_MODE = [ALL, SUMMARY, MD5]

    WRITE_FLAGS = os.O_WRONLY | os.O_CREAT
    WRITE_MODES = stat.S_IWUSR | stat.S_IRUSR
    OVERWRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_TRUNC

    PKL_SUFFIX = ".pkl"
    NUMPY_SUFFIX = ".npy"
    PT_SUFFIX = ".pt"
    ONE_GB = 1073741824  # 1 * 1024 * 1024 * 1024
    TEN_GB = 10737418240  # 10 * 1024 * 1024 * 1024
    ONE_MB = 1048576  # 1 * 1024 * 1024
    FILE_PATTERN = r'^[a-zA-Z0-9_./-]+$'
    DISTRIBUTED_PREFIX_LENGTH = 60
    # env dump path
    KWARGS = 'kwargs'
    INPUT = 'input'
    OUTPUT = 'output'
    INPUT_ARGS = 'input_args'
    INPUT_KWARGS = 'input_kwargs'
    GRAD_INPUT = 'grad_input'
    GRAD_OUTPUT = 'grad_output'
    START = "start"
    STOP = "stop"
    ENV_ENABLE = "1"
    ENV_DISABLE = "0"
    MAX_SEED_VALUE = 4294967295  # 2**32 - 1
    STATISTICS = "statistics"
    TENSOR = "tensor"
    OVERFLOW_CHECK = "overflow_check"
    FREE_BENCHMARK = "free_benchmark"
    RUN_UT = "run_ut"
    GRAD_PROBE = "grad_probe"
    TASK_LIST = [TENSOR, STATISTICS, OVERFLOW_CHECK, FREE_BENCHMARK, RUN_UT, GRAD_PROBE]
    DUMP_DATA_COLLECTION_LIST = [STATISTICS, TENSOR]
    DUMP_DATA_MODE_LIST = [ALL, INPUT, OUTPUT, FORWARD, BACKWARD]
    LEVEL_L0 = "L0"
    LEVEL_L1 = "L1"
    LEVEL_L2 = "L2"
    LEVEL_MIX = "mix"
    LEVEL_LIST = [LEVEL_L0, LEVEL_L1, LEVEL_L2, LEVEL_MIX]
    ATTR_NAME_PREFIX = "wrap_"
    ATTR_NAME_PREFIX_LEN = len(ATTR_NAME_PREFIX)
    KERNEL_DUMP = "kernel_dump"
    DATA = "data"
    PT_FRAMEWORK = "pytorch"
    MS_FRAMEWORK = "mindspore"
    DIRECTORY_LENGTH = 4096
    FILE_NAME_LENGTH = 255
    FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble, np.float32, np.float16]
    BOOL_TYPE = [bool, np.uint8]
    INT_TYPE = [np.int32, np.int64]
    NPU = 'NPU'
    NPU_LOWERCASE = 'npu'
    CPU_LOWERCASE = 'cpu'
    CUDA_LOWERCASE = 'cuda'
    DISTRIBUTED = 'Distributed'

    # struct json param
    ORIGIN_DATA = "origin_data"
    SCOPE = "scope"
    STACK = "stack"

    ATEN = "Aten"
    MODULE_WHITE_LIST = ["torch", "numpy"]

    FUNC_SKIP_LIST = ["construct", "__call__"]

    FILE_SKIP_LIST = ["site-packages/mindspore", "package/mindspore", "msprobe", "site-packages/torch", "package/torch"]

    STACK_FILE_INDEX = 0

    STACK_FUNC_INDEX = 2

    STACK_FUNC_ELE_INDEX = 1

    CONSTRUCT_NAME_INDEX = -3

    NAME_FIRST_POSSIBLE_INDEX = -4

    NAME_SECOND_POSSIBLE_INDEX = -5

    INPLACE_LIST = [
        "broadcast", "all_reduce", "reduce", "all_gather", "gather", "scatter", "reduce_scatter",
        "_reduce_scatter_base", "_all_gather_base", "send", "recv", "irecv", "isend", "all_to_all_single", "all_to_all",
        "all_gather_into_tensor", "reduce_scatter_tensor"
    ]

    CONVERT = {
        "int32_to_int64": ["torch.int32", "torch.int64"],
    }

    CONVERT_API = {
        "int32_to_int64": ["cross_entropy"]
    }

    FILL_CHAR_NUMS = 50
    TOOL_ENDS_SUCCESSFULLY = f"{TOOL_NAME} ends successfully."
    WITHOUT_CALL_STACK = "The call stack retrieval failed."

    STEP = "step"
    RANK = "rank"
    HYPHEN = "-"
    STEP_RANK_MAXIMUM_RANGE = [int(0), int(1e6)]

    # data type const
    FLOAT16 = "Float16"
    FLOAT32 = "Float32"
    BFLOAT16 = "BFloat16"
    TORCH_FLOAT16 = "torch.float16"
    TORCH_FLOAT32 = "torch.float32"
    TORCH_BFLOAT16 = "torch.bfloat16"

    DTYPE = 'dtype'
    SHAPE = 'shape'
    MAX = 'Max'
    MIN = 'Min'
    MEAN = 'Mean'
    NORM = 'Norm'


class CompareConst:
    """
    Class for compare module const
    """
    SPACE = " "
    # compare result column name
    NPU_NAME = "NPU Name"
    BENCH_NAME = "Bench Name"
    NPU_DTYPE = "NPU Dtype"
    BENCH_DTYPE = "Bench Dtype"
    NPU_SHAPE = "NPU Tensor Shape"
    BENCH_SHAPE = "Bench Tensor Shape"
    NPU_MAX = "NPU max"
    NPU_MIN = "NPU min"
    NPU_MEAN = "NPU mean"
    NPU_NORM = "NPU l2norm"
    BENCH_MAX = "Bench max"
    BENCH_MIN = "Bench min"
    BENCH_MEAN = "Bench mean"
    BENCH_NORM = "Bench l2norm"
    MAX_DIFF = "Max diff"
    MIN_DIFF = "Min diff"
    MEAN_DIFF = "Mean diff"
    NORM_DIFF = "L2norm diff"
    COSINE = "Cosine"
    MAX_ABS_ERR = "MaxAbsErr"
    MAX_RELATIVE_ERR = "MaxRelativeErr"
    MIN_RELATIVE_ERR = "MinRelativeErr"
    MEAN_RELATIVE_ERR = "MeanRelativeErr"
    NORM_RELATIVE_ERR = "NormRelativeErr"
    ACCURACY = "Accuracy Reached or Not"
    STACK = "NPU_Stack_Info"
    DATA_NAME = "Data_name"
    ERROR_MESSAGE = "Err_message"
    ONE_THOUSANDTH_ERR_RATIO = "One Thousandth Err Ratio"
    FIVE_THOUSANDTHS_ERR_RATIO = "Five Thousandths Err Ratio"
    NPU_MD5 = "NPU MD5"
    BENCH_MD5 = "BENCH MD5"
    RESULT = "Result"
    MAGNITUDE = 0.5
    OP_NAME = "op_name"
    INPUT_STRUCT = "input_struct"
    KWARGS_STRUCT = "kwargs_struct"
    OUTPUT_STRUCT = "output_struct"
    SUMMARY = "summary"
    MAX_EXCEL_LENGTH = 1048576
    YES = "Yes"
    NO = "No"
    STATISTICS_INDICATOR_NUM = 4
    EPSILON = 1e-10

    COMPARE_RESULT_HEADER = [
        NPU_NAME, BENCH_NAME, NPU_DTYPE, BENCH_DTYPE, NPU_SHAPE, BENCH_SHAPE, COSINE, MAX_ABS_ERR, MAX_RELATIVE_ERR,
        ONE_THOUSANDTH_ERR_RATIO, FIVE_THOUSANDTHS_ERR_RATIO,
        NPU_MAX, NPU_MIN, NPU_MEAN, NPU_NORM, BENCH_MAX, BENCH_MIN, BENCH_MEAN, BENCH_NORM, ACCURACY, ERROR_MESSAGE
    ]

    SUMMARY_COMPARE_RESULT_HEADER = [
        NPU_NAME, BENCH_NAME, NPU_DTYPE, BENCH_DTYPE, NPU_SHAPE, BENCH_SHAPE, MAX_DIFF, MIN_DIFF, MEAN_DIFF, NORM_DIFF,
        MAX_RELATIVE_ERR, MIN_RELATIVE_ERR, MEAN_RELATIVE_ERR, NORM_RELATIVE_ERR,
        NPU_MAX, NPU_MIN, NPU_MEAN, NPU_NORM, BENCH_MAX, BENCH_MIN, BENCH_MEAN, BENCH_NORM, RESULT, ERROR_MESSAGE
    ]

    MD5_COMPARE_RESULT_HEADER = [
        NPU_NAME, BENCH_NAME, NPU_DTYPE, BENCH_DTYPE, NPU_SHAPE, BENCH_SHAPE, NPU_MD5, BENCH_MD5, RESULT
    ]

    HEAD_OF_COMPARE_MODE = {
        Const.ALL: COMPARE_RESULT_HEADER,
        Const.SUMMARY: SUMMARY_COMPARE_RESULT_HEADER,
        Const.MD5: MD5_COMPARE_RESULT_HEADER
    }

    # compare standard
    HUNDRED_RATIO_THRESHOLD = 0.01
    THOUSAND_RATIO_THRESHOLD = 0.001
    FIVE_THOUSAND_RATIO_THRESHOLD = 0.005
    TEN_THOUSAND_RATIO_THRESHOLD = 0.0001
    COSINE_THRESHOLD = 0.9999
    ULP_FLOAT32_THRESHOLD = 32
    ULP_FLOAT16_THRESHOLD = 1

    # compare result data
    READ_NONE = 'No data'
    NONE = 'None'
    SHAPE_UNMATCH = 'shape unmatched'
    DIFF = 'Different'
    UNSUPPORTED = 'unsupported'
    NAN = 'Nan'
    PASS = 'pass'
    WARNING = 'Warning'
    ERROR = 'error'
    TRUE = 'TRUE'
    FALSE = 'FALSE'
    SKIP = 'SKIP'
    N_A = 'N/A'
    INF = 'inf'
    NEG_INF = '-inf'
    BFLOAT16_MIN = -3.3895313892515355e+38
    BFLOAT16_MAX = 3.3895313892515355e+38
    BFLOAT16_EPS = 3.90625e-3  # 2 ** -8

    # accuracy standards
    COS_THRESHOLD = 0.99
    MAX_ABS_ERR_THRESHOLD = 0.001
    MAX_RELATIVE_ERR_THRESHOLD = 0.001
    COS_MAX_THRESHOLD = 0.9
    MAX_ABS_ERR_MAX_THRESHOLD = 1
    ACCURACY_CHECK_YES = "Yes"
    ACCURACY_CHECK_NO = "No"
    ACCURACY_CHECK_UNMATCH = "Unmatched"

    # error message
    NO_BENCH = "No bench data matched."

    # compare const
    FLOAT_TYPE = [np.half, np.single, float, np.double, np.float64, np.longdouble]

    # highlight xlsx color const
    RED = "FFFF0000"
    YELLOW = "FFFF00"
    BLUE = "0000FF"

    # run_ut const
    MAX_TOKENS = 65536
    SPECIAL_SPARSE_MOED = 4

    # highlight rules const
    OVERFLOW_LIST = ['nan\t', 'inf\t', '-inf\t', 'nan', 'inf', '-inf']
    MAX_DIFF_RED = 1e+10
    ORDER_MAGNITUDE_DIFF_YELLOW = 1
    ONE_THOUSAND_ERROR_IN_RED = 0.9
    ONE_THOUSAND_ERROR_OUT_RED = 0.6
    ONE_THOUSAND_ERROR_DIFF_YELLOW = 0.1
    COSINE_DIFF_YELLOW = 0.1
    MAX_RELATIVE_OUT_RED = 0.5
    MAX_RELATIVE_OUT_YELLOW = 0.1
    MAX_RELATIVE_IN_YELLOW = 0.01
    MS_GRAPH_BASE = {
        NPU_NAME: None, BENCH_NAME: None, NPU_DTYPE: None, BENCH_DTYPE: None, NPU_SHAPE: None, BENCH_SHAPE: None,
        NPU_MAX: None, NPU_MIN: None, NPU_MEAN: None, NPU_NORM: None, BENCH_MAX: None, BENCH_MIN: None,
        BENCH_MEAN: None, BENCH_NORM: None, ACCURACY: '', ERROR_MESSAGE: ''
    }
    MS_GRAPH_NPY = {
        COSINE: None, MAX_ABS_ERR: None, MAX_RELATIVE_ERR: None, ONE_THOUSANDTH_ERR_RATIO: None,
        FIVE_THOUSANDTHS_ERR_RATIO: None
    }
    MS_GRAPH_STATISTIC = {
        MAX_DIFF: None, MIN_DIFF: None, MEAN_DIFF: None, NORM_DIFF: None, MAX_RELATIVE_ERR: None,
        MIN_RELATIVE_ERR: None, MEAN_RELATIVE_ERR: None, NORM_RELATIVE_ERR: None
    }

DETAIL_TEST_ROWS = [
            [
            "API Name", "Bench Dtype", "DEVICE Dtype", "Shape",
            "余弦相似度",
            "最大绝对误差",
            "双百指标",
            "双千指标",
            "双万指标",
            "二进制一致错误率",
            "误差均衡性",
            "均方根误差",
            "小值域错误占比",
            "相对误差最大值",
            "相对误差平均值",
            "inf/nan错误率",
            "相对误差错误率",
            "绝对误差错误率",
            "ULP误差最大值",
            "ULP误差平均值",
            "ULP误差大于阈值占比",
            "Status",
            "Message"
            ]
        ]


precision_configs = {
    torch.float16 : {
        'small_value' : [
            1e-3
        ],
        'small_value_atol' : [
            1e-5
        ]
    },
    torch.bfloat16: {
        'small_value' : [
            1e-3
        ],
        'small_value_atol' : [
            1e-5
        ]
    },
    torch.float32:{
        'small_value' : [
            1e-6
        ],
        'small_value_atol' : [
            1e-9
        ]
    },
    torch.float8_e4m3fn:{
        'small_value' : [
            1e-6
        ],
        'small_value_atol' : [
            1e-9
        ]
    }
}


ULP_PARAMETERS = {
    torch.float16 : {
        'min_eb' : [
            -14
        ],
        'mantissa_num' : [
            10
        ]
    },
    torch.bfloat16 : {
        'min_eb' : [
            -126
        ],
        'mantissa_num' : [
            7
        ]
    },
    torch.float32 : {
        'min_eb' : [
            -126
        ],
        'mantissa_num' : [
            23
        ]
    },
    torch.float8_e4m3fn:{
        'min_eb' : [
            -6
        ],
        'mantissa_num' : [
            3
        ]
    }
}


class ApiPrecisionCompareColumn:
    API_NAME = 'API Name'
    DEVICE_DTYPE = 'DEVICE Dtype'
    SMALL_VALUE_ERROR_RATE = '小值域错误占比'
    RMSE = '均方根误差'
    MAX_REL_ERR = '相对误差最大值'
    MEAN_REL_ERR = '相对误差平均值'
    EB = '误差均衡性'
    SMALL_VALUE_ERROR_RATIO = '小值域错误比值'
    SMALL_VALUE_ERROR_STATUS = '小值域判定结果'
    RMSE_RATIO = '均方根误差比值'
    RMSE_STATUS = '均方根误差判定结果'
    MAX_REL_ERR_RATIO = '相对误差最大值比值'
    MAX_REL_ERR_STATUS = '相对误差最大值判定结果'
    MEAN_REL_ERR_RATIO = '相对误差平均值比值'
    MEAN_REL_ERR_STATUS = '相对误差平均值判定结果'
    EB_RATIO = '误差均衡性比值'
    EB_STATUS = '误差均衡性判定结果'
    ERROR_RATE = '二进制一致错误率'
    ERROR_RATE_STATUS = '二进制一致错误率判定结果'
    INF_NAN_ERROR_RATIO = 'inf/nan错误率'
    INF_NAN_ERROR_RATIO_STATUS = 'inf/nan判定结果'
    REL_ERR_RATIO = '相对误差错误率'
    REL_ERR_RATIO_STATUS = '相对误差判定结果'
    ABS_ERR_RATIO = '绝对误差错误率'
    ABS_ERR_RATIO_STATUS = '绝对误差判定结果'
    MEAN_ULP_ERR = 'ULP误差平均值'
    ULP_ERR_PROPORTION = 'ULP误差大于阈值占比'
    ULP_ERR_PROPORTION_RATIO = 'ULP误差大于阈值占比比值'
    ULP_ERR_STATUS = 'ULP误差判定结果'
    REL_ERR_THOUSANDTH = '双千指标'
    REL_ERR_THOUSANDTH_STATUS = '双千指标判定结果'
    FINAL_RESULT = '比对结果'
    ALGORITHM = '比对算法'
    FORWWARD_STATUS = 'Forward Test Success'
    BACKWARD_STATUS = 'Backward Test Success'
    MESSAGE = 'Message'

    @staticmethod
    def to_required_columns():
        return [ApiPrecisionCompareColumn.API_NAME, ApiPrecisionCompareColumn.DEVICE_DTYPE, 
                ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATE, ApiPrecisionCompareColumn.RMSE, 
                ApiPrecisionCompareColumn.MAX_REL_ERR, ApiPrecisionCompareColumn.MEAN_REL_ERR,
                ApiPrecisionCompareColumn.EB, ApiPrecisionCompareColumn.ERROR_RATE, 
                ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO, ApiPrecisionCompareColumn.REL_ERR_RATIO, 
                ApiPrecisionCompareColumn.ABS_ERR_RATIO, ApiPrecisionCompareColumn.MEAN_ULP_ERR, 
                ApiPrecisionCompareColumn.ULP_ERR_PROPORTION, ApiPrecisionCompareColumn.REL_ERR_THOUSANDTH]

    @staticmethod
    def get_detail_csv_title():
        return [ApiPrecisionCompareColumn.API_NAME, 
                ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_RATIO, ApiPrecisionCompareColumn.SMALL_VALUE_ERROR_STATUS, 
                ApiPrecisionCompareColumn.RMSE_RATIO, ApiPrecisionCompareColumn.RMSE_STATUS, 
                ApiPrecisionCompareColumn.MAX_REL_ERR_RATIO, ApiPrecisionCompareColumn.MAX_REL_ERR_STATUS, 
                ApiPrecisionCompareColumn.MEAN_REL_ERR_RATIO, ApiPrecisionCompareColumn.MEAN_REL_ERR_STATUS, 
                ApiPrecisionCompareColumn.EB_RATIO, ApiPrecisionCompareColumn.EB_STATUS, 
                ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO, ApiPrecisionCompareColumn.INF_NAN_ERROR_RATIO_STATUS, 
                ApiPrecisionCompareColumn.REL_ERR_RATIO, ApiPrecisionCompareColumn.REL_ERR_RATIO_STATUS, 
                ApiPrecisionCompareColumn.ABS_ERR_RATIO, ApiPrecisionCompareColumn.ABS_ERR_RATIO_STATUS, 
                ApiPrecisionCompareColumn.ERROR_RATE, ApiPrecisionCompareColumn.ERROR_RATE_STATUS, 
                ApiPrecisionCompareColumn.MEAN_ULP_ERR, ApiPrecisionCompareColumn.ULP_ERR_PROPORTION, 
                ApiPrecisionCompareColumn.ULP_ERR_PROPORTION_RATIO, ApiPrecisionCompareColumn.ULP_ERR_STATUS,
                ApiPrecisionCompareColumn.REL_ERR_THOUSANDTH, ApiPrecisionCompareColumn.REL_ERR_THOUSANDTH_STATUS,
                ApiPrecisionCompareColumn.FINAL_RESULT, ApiPrecisionCompareColumn.ALGORITHM, 
                ApiPrecisionCompareColumn.MESSAGE]

    @staticmethod
    def get_result_csv_title():
        return [ApiPrecisionCompareColumn.API_NAME, ApiPrecisionCompareColumn.FORWWARD_STATUS, 
                ApiPrecisionCompareColumn.BACKWARD_STATUS, ApiPrecisionCompareColumn.MESSAGE]


CompareMessage = {
    "topk" : "在npu上，topk的入参sorted=False时不生效，会返回有序tensor，而cpu上会返回无序tensor。 如果topk精度不达标，请检查是否是该原因导致的。"
}


def check_dtype_comparable(x, y):
    if x.dtype in Const.FLOAT_TYPE:
        if y.dtype in Const.FLOAT_TYPE:
            return True 
        return False 
    if x.dtype in Const.BOOL_TYPE:
        if y.dtype in Const.BOOL_TYPE:
            return True 
        return False 
    if x.dtype in Const.INT_TYPE:
        if y.dtype in Const.INT_TYPE:
            return True 
        return False
    print(f"Compare: Unexpected dtype {x.dtype}, {y.dtype}")
    return False


def convert_str_to_float(input_data):
    if isinstance(input_data, str) and input_data.strip() == "":
        msg = 'ERROR: Input data is an empty string'
    try:
        float_data = float(input_data)
        return float_data
    except ValueError as e:
        msg = 'ERROR: Input data cannot be converted to float'
        print(msg)


def is_inf_or_nan(x):
    return math.isnan(x) or math.isinf(x)


def handle_infinity(x, y, column_name):
    if math.isinf(x) and math.isinf(y):
        if x == y:
            return float("nan"), True, f"{column_name}同为同号inf或nan\n"
        else:
            return float("nan"), False, f"{column_name}inf或nan不一致\n"
    else:
        return float("nan"), False, f"{column_name}inf或nan不一致\n"


def handle_nan(x, y, column_name):
    if math.isnan(x) and math.isnan(y):
        return float("nan"), True, f"{column_name}同为同号inf或nan\n"
    else:
        return float("nan"), False, f"{column_name}inf或nan不一致\n"


def check_inf_or_nan(x, y, column_name):
    if math.isinf(x) or math.isinf(y):
        return handle_infinity(x, y, column_name)
    else:
        return handle_nan(x, y, column_name)
