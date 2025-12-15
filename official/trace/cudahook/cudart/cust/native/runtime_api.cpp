#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_API_PER_THREAD_DEFAULT_STREAM
#define __CUDA_API_VERSION_INTERNAL

// auto generate 2 apis

#include "cuda_runtime_api.h"
#include "common/hook.h"
#include "common/macro_common.h"
#include "common/api_log.h"
#include "common/record.h"
#include "common/utils.hpp"
#include "cudart/cust/declares.h"
#include "cudart/cust/common.h"

namespace hook {
namespace native {
cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
  using func_ptr =
      cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaLaunchKernel"));
  HOOK_CHECK(func_entry);
  auto ret = func_entry(func, gridDim, blockDim, args, sharedMem, stream);
  DumpKernelName(func);
  return ret;
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                const void *func, void **args) {
  using func_ptr =
      cudaError_t (*)(const cudaLaunchConfig_t *, const void *, void **);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaLaunchKernelExC"));
  HOOK_CHECK(func_entry);
  auto ret = func_entry(config, func, args);
  DumpKernelName(func);
  return ret;
}

cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3 gridDim,
                                        dim3 blockDim, void **args,
                                        size_t sharedMem, cudaStream_t stream) {
  using func_ptr =
      cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDART_SYMBOL("cudaLaunchCooperativeKernel"));
  HOOK_CHECK(func_entry);
  auto ret = func_entry(func, gridDim, blockDim, args, sharedMem, stream);
  DumpKernelName(func);
  return ret;
}

cudaError_t cudaSetDevice(int device) {
  using func_ptr = cudaError_t (*)(int);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaSetDevice"));
  HOOK_CHECK(func_entry);
  auto ret = func_entry(device);
  setDeviceId(device);
  return ret;
}

cudaError_t cudaGetDriverEntryPoint(
    const char *symbol, void **funcPtr, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *driverStatus) {
  return hookCudaGetDriverEntryPoint(symbol, funcPtr, flags, driverStatus);
}

NATIVE_REGISTE_FUNC(cudaGetDriverEntryPoint)

NATIVE_REGISTE_FUNC(cudaLaunchKernel)
NATIVE_REGISTE_FUNC(cudaLaunchKernelExC)
NATIVE_REGISTE_FUNC(cudaLaunchCooperativeKernel)
NATIVE_REGISTE_FUNC(cudaSetDevice)

}  // namespace native
}  // namespace hook
