#pragma once

#include "common/error_check_base.h"
#include "header/include/nccl.h"

namespace {
template <>
bool CheckError<ncclResult_t>(ncclResult_t x) {
  return ncclSuccess == x;
}
}  // namespace
