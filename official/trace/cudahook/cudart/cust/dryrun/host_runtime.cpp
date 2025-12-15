#define __CUDACC_RTC__
#define __CUDA_API_PER_THREAD_DEFAULT_STREAM
#define __CUDA_API_VERSION_INTERNAL

// auto generate 12 apis
#include <cstddef>
#include "crt/host_runtime.h"
#include "common/hook.h"
#include "common/macro_common.h"
#include "common/log.h"
// #include "common/black_list.h"
#include "common/record.h"
#include "common/utils.hpp"
#include "cudart/cust/declares.h"
#include "cudart/cust/dryrun/callCfgStack.h"

namespace hook {
namespace dryrun {
using myFatbin = int;
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

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                       size_t *sharedMem, void *stream) {
  using func_ptr = cudaError_t (*)(dim3 *, dim3 *, size_t *, void *);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDART_SYMBOL("__cudaPopCallConfiguration"));
  HOOK_CHECK(func_entry);
  return func_entry(gridDim, blockDim, sharedMem, stream);
}

void **__cudaRegisterFatBinary(void *fatCubin) {
  using func_ptr = void **(*)(void *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaRegisterFatBinary"));
  return func_entry(fatCubin);
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle) {
  using func_ptr = void (*)(void **);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDART_SYMBOL("__cudaRegisterFatBinaryEnd"));
  return func_entry(fatCubinHandle);
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar,
                       char *deviceAddress, const char *deviceName, int ext,
                       size_t size, int constant, int global) {
  using func_ptr =
      void (*)(void **, char *, char *, const char *, int, size_t, int, int);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("__cudaRegisterVar"));
  HOOK_CHECK(func_entry);
  return func_entry(fatCubinHandle, hostVar, deviceAddress, deviceName, ext,
                    size, constant, global);
}

void __cudaUnregisterFatBinary(void **fatCubinHandle) {
  using func_ptr = void (*)(void **);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDART_SYMBOL("__cudaUnregisterFatBinary"));
  HOOK_CHECK(func_entry);
  return func_entry(fatCubinHandle);
}

DRYRUN_REGISTE_FUNC(__cudaRegisterFunction)
DRYRUN_REGISTE_FUNC(__cudaPopCallConfiguration)
DRYRUN_REGISTE_FUNC(__cudaRegisterFatBinary)
DRYRUN_REGISTE_FUNC(__cudaRegisterFatBinaryEnd)
DRYRUN_REGISTE_FUNC(__cudaRegisterVar)
DRYRUN_REGISTE_FUNC(__cudaUnregisterFatBinary)
}  // namespace dryrun
}  // namespace hook
