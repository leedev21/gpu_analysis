#include <cstdint>

#include "common/error_string.h"
#include "common/hook.h"
#include "common/macro_common.h"

namespace hook {
const char* getErrorString(cudaError_t error) {
  using func_ptr = const char* (*)(cudaError_t);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetErrorString"));
  HOOK_CHECK(func_entry);
  return func_entry(error);
}
}  // namespace hook
