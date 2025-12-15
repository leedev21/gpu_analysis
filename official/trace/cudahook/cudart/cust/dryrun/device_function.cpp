#define __CUDACC_RTC__
#define __CUDA_API_PER_THREAD_DEFAULT_STREAM
#define __CUDA_API_VERSION_INTERNAL

// auto generate 4 apis
#include <cstddef>
#include "crt/device_functions.h"
#include "common/hook.h"
#include "common/macro_common.h"
#include "common/api_log.h"
#include "common/utils.hpp"
#include "cudart/cust/declares.h"
#include "cudart/cust/dryrun/callCfgStack.h"

namespace hook {
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem,
                                     struct CUstream_st *stream) {
  using func_ptr =
      __device__ unsigned (*)(dim3, dim3, size_t, struct CUstream_st *);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDART_SYMBOL("__cudaPushCallConfiguration"));
  HOOK_CHECK(func_entry);
  return func_entry(gridDim, blockDim, sharedMem, stream);
}

DRYRUN_REGISTE_FUNC(__cudaPushCallConfiguration)

}  // namespace hook
