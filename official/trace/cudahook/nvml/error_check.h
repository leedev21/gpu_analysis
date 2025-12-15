#pragma once

#include "common/error_check_base.h"
#include "nvml.h"

namespace {
template <>
bool CheckError<nvmlReturn_t>(nvmlReturn_t x) {
  return NVML_SUCCESS == x;
}
}  // namespace
