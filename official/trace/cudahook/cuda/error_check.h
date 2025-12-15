#pragma once

#include "common/error_check_base.h"
#include "cuda.h"

namespace {
template <>
bool CheckError<CUresult>(CUresult x) {
  if (CUDA_SUCCESS != x) {
    const char* error = nullptr;
    cuGetErrorString(x, &error);
    HLOGEX("%s error %d : %s\n", __func__, x, error);
    return false;
  } else {
    return true;
  }
}
}  // namespace
