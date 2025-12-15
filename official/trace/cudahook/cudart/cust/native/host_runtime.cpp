#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_API_PER_THREAD_DEFAULT_STREAM
#define __CUDA_API_VERSION_INTERNAL

// auto generate 2 apis
#include <cstddef>
#include "crt/host_runtime.h"
#include "common/hook.h"
#include "common/macro_common.h"
#include "common/api_log.h"
#include "common/record.h"
#include "common/utils.hpp"
#include "cudart/cust/declares.h"

namespace hook {
namespace native {
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *bid,
                            dim3 *bDim, dim3 *gDim, int *wSize) {
  using func_ptr = void (*)(void **, const char *, char *, const char *, int,
                            uint3 *, uint3 *, dim3 *, dim3 *, int *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaRegisterFunction"));
  HOOK_CHECK(func_entry);
  GetKernelNameStore()->Registe(hostFun, deviceFun);
  return func_entry(fatCubinHandle, hostFun, deviceFun, deviceName,
                    thread_limit, tid, bid, bDim, gDim, wSize);
}
NATIVE_REGISTE_FUNC(__cudaRegisterFunction)
}  // namespace native
}  // namespace hook
