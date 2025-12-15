#pragma once
#include "common/macro_common.h"
#include "common/log.h"

namespace {
template <typename T>
bool CheckError(T) {
  return false;
}

template <>
bool CheckError<const char *>(const char *x) {
  return nullptr != x;
}

template <>
bool CheckError<bool>(bool x) {
  return x;
}
}  // namespace

#define CHECK_RETURN(expr)                                      \
  do {                                                          \
    if (!CheckError(expr)) {                                    \
      HLOGEX("Check failed %s for %s %s %d\n", #expr, __func__, \
             HOOK_LOG_FILE(__FILE__), __LINE__);                \
      exit(EXIT_FAILURE);                                       \
    }                                                           \
  } while (0)
