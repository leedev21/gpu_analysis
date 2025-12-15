#pragma once

#include "common/error_check_base.h"
#include "cuda_runtime_api.h"
#include "common/error_string.h"

namespace {
template <>
bool CheckError<cudaError_t>(cudaError_t x) {
  if (cudaSuccess != x) {
    HLOGEX("%s %s", __func__, hook::getErrorString(x));
    return false;
  } else {
    return true;
  }
}
}  // namespace
