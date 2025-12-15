#include <dlfcn.h>

#include "cuda_runtime_api.h"
#include "common/macro_common.h"

namespace hook {
cudaError_t hookCudaGetDriverEntryPoint(
    const char *symbol, void **funcPtr, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *driverStatus) {
  static auto drverHandle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  HOOK_CHECK(drverHandle);
  auto fn = dlsym(drverHandle, symbol);
  cudaError_t ret;
  if (!fn) {
    *driverStatus = cudaDriverEntryPointSymbolNotFound;
    ret = cudaErrorInvalidValue;
  } else {
    *funcPtr = fn;
    *driverStatus = cudaDriverEntryPointSuccess;
    ret = cudaSuccess;
  }
  return ret;
}
}  // namespace hook
