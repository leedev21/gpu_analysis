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
#include "cudart/cust/dryrun/device_prop.h"
#include "cudart/cust/common.h"

namespace hook {
namespace dryrun {
using DryrunEvent = int;
using DryrunStream = int;

cudaError_t cudaLaunchKernel(const void *func, dim3, dim3, void **, size_t,
                             cudaStream_t) {
  DumpKernelName(func);
  return cudaSuccess;
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *, const void *func,
                                void **) {
  DumpKernelName(func);
  return cudaSuccess;
}

cudaError_t cudaLaunchCooperativeKernel(const void *func, dim3, dim3, void **,
                                        size_t, cudaStream_t) {
  DumpKernelName(func);
  return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }

cudaError_t cudaMemcpyAsync(void *, const void *, size_t, enum cudaMemcpyKind,
                            cudaStream_t) {
  return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t) {
  return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned int) {
  return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return cudaSuccess; }

cudaError_t cudaMemcpy(void *, const void *, size_t, enum cudaMemcpyKind) {
  return cudaSuccess;
}

cudaError_t cudaStreamIsCapturing(
    cudaStream_t, enum cudaStreamCaptureStatus *pCaptureStatus) {
  if (pCaptureStatus) {
    *pCaptureStatus = cudaStreamCaptureStatusNone;
  }
  return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t *event) {
  if (event) {
    *event = reinterpret_cast<cudaEvent_t>(new DryrunEvent);
  }
  return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int) {
  if (event) {
    *event = reinterpret_cast<cudaEvent_t>(new DryrunEvent);
  }
  return cudaSuccess;
}
cudaError_t cudaEventDestroy(cudaEvent_t event) {
  if (event) {
    delete reinterpret_cast<DryrunEvent *>(event);
  }
  return cudaSuccess;
}
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t, cudaEvent_t) {
  if (ms) {
    *ms = 1;
  }
  return cudaSuccess;
}
cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  *devPtr = malloc(size);
  return cudaSuccess;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int) {
  *pHost = malloc(size);
  return cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
  free(devPtr);
  return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop, int id) {
  QueryDeviceProp(prop, id);
  return cudaSuccess;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *p, unsigned int, int) {
  if (p) {
    *p = reinterpret_cast<cudaStream_t>(new DryrunStream);
  }
  return cudaSuccess;
}

cudaError_t cudaStreamCreate(cudaStream_t *p) {
  if (p) {
    *p = reinterpret_cast<cudaStream_t>(new DryrunStream);
  }
  return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *p, unsigned int) {
  if (p) {
    *p = reinterpret_cast<cudaStream_t>(new DryrunStream);
  }
  return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  if (stream) {
    delete reinterpret_cast<DryrunStream *>(stream);
  }
  return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
cudaError_t cudaDeviceSynchronize(cudaEvent_t) { return cudaSuccess; }

cudaError_t cudaGetDeviceCount(int *count) {
  using func_ptr = cudaError_t (*)(int *);
  static auto fn =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetDeviceCount"));
  HOOK_CHECK(fn);
  auto ret = fn(count);
  if (cudaSuccess != ret) {
    HLOGEX("error, %s return %s", __func__, ret);
  }

  *count = getFakeDeviceCount();
  return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
  using func_ptr = cudaError_t (*)(int);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaSetDevice"));
  HOOK_CHECK(func_entry);
  auto ret = func_entry(0);
  if (cudaSuccess != ret) {
    HLOGEX("error, %s return %s", __func__, ret);
  }

  setDeviceId(device);
  return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device) {
  using func_ptr = cudaError_t (*)(int *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetDevice"));
  HOOK_CHECK(func_entry);
  auto ret = func_entry(device);
  if (cudaSuccess != ret) {
    HLOGEX("error, %s return %s", __func__, ret);
  }

  *device = getDeviceId();
  return cudaSuccess;
}

DRYRUN_REGISTE_FUNC2(cudaLaunchKernel)
DRYRUN_REGISTE_FUNC2(cudaLaunchKernelExC)
DRYRUN_REGISTE_FUNC2(cudaLaunchCooperativeKernel)
DRYRUN_REGISTE_FUNC2(cudaStreamSynchronize)
DRYRUN_REGISTE_FUNC2(cudaMemcpyAsync)
DRYRUN_REGISTE_FUNC2(cudaMemsetAsync)
DRYRUN_REGISTE_FUNC2(cudaStreamWaitEvent)
DRYRUN_REGISTE_FUNC2(cudaEventRecord)
DRYRUN_REGISTE_FUNC2(cudaMemcpy)
DRYRUN_REGISTE_FUNC2(cudaStreamIsCapturing)
DRYRUN_REGISTE_FUNC2(cudaEventCreate)
DRYRUN_REGISTE_FUNC2(cudaEventCreateWithFlags)
DRYRUN_REGISTE_FUNC2(cudaEventDestroy)
DRYRUN_REGISTE_FUNC2(cudaEventElapsedTime)
DRYRUN_REGISTE_FUNC2(cudaEventQuery)
DRYRUN_REGISTE_FUNC2(cudaMalloc)
DRYRUN_REGISTE_FUNC2(cudaHostAlloc)
DRYRUN_REGISTE_FUNC2(cudaFree)
DRYRUN_REGISTE_FUNC2(cudaGetDeviceProperties_v2)
DRYRUN_REGISTE_FUNC2(cudaStreamDestroy)
DRYRUN_REGISTE_FUNC2(cudaDeviceSynchronize)
DRYRUN_REGISTE_FUNC2(cudaEventSynchronize)
DRYRUN_REGISTE_FUNC2(cudaStreamCreateWithPriority)
DRYRUN_REGISTE_FUNC2(cudaStreamCreate)
DRYRUN_REGISTE_FUNC2(cudaStreamCreateWithFlags)
DRYRUN_REGISTE_FUNC2(cudaGetDeviceCount)
DRYRUN_REGISTE_FUNC2(cudaSetDevice)
DRYRUN_REGISTE_FUNC2(cudaGetDevice)

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    int *numBlocks, const void *func, int blockSize, size_t dynamicSMemSize,
    unsigned int flags) {
  using func_ptr =
      cudaError_t (*)(int *, const void *, int, size_t, unsigned int);
  static auto func_entry = reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL(
      "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"));
  auto _ret = func_entry(numBlocks, func, blockSize, dynamicSMemSize, flags);
  return _ret;
}

cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority) {
  using func_ptr = cudaError_t (*)(int *, int *);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDART_SYMBOL("cudaDeviceGetStreamPriorityRange"));
  auto _ret = func_entry(leastPriority, greatestPriority);
  return _ret;
}

cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr,
                                   int device) {
  using func_ptr = cudaError_t (*)(int *, enum cudaDeviceAttr, int);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaDeviceGetAttribute"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(value, attr, 0);
  return _ret;
}

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr,
                                  const void *func) {
  using func_ptr = cudaError_t (*)(struct cudaFuncAttributes *, const void *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaFuncGetAttributes"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(attr, func);
  return _ret;
}

cudaError_t cudaFuncSetAttribute(const void *func, enum cudaFuncAttribute attr,
                                 int value) {
  using func_ptr = cudaError_t (*)(const void *, enum cudaFuncAttribute, int);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaFuncSetAttribute"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(func, attr, value);
  return _ret;
}

cudaError_t cudaGetLastError() {
  using func_ptr = cudaError_t (*)();
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetLastError"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry();
  return _ret;
}

cudaError_t cudaPeekAtLastError() {
  using func_ptr = cudaError_t (*)();
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaPeekAtLastError"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry();
  return _ret;
}

cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,
                                     const void *ptr) {
  using func_ptr =
      cudaError_t (*)(struct cudaPointerAttributes *, const void *);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDART_SYMBOL("cudaPointerGetAttributes"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(attributes, ptr);
  return _ret;
}

cudaError_t cudaGetDriverEntryPoint(
    const char *symbol, void **funcPtr, unsigned long long flags,
    enum cudaDriverEntryPointQueryResult *driverStatus) {
  using func_ptr = cudaError_t (*)(const char *, void **, unsigned long long,
                                   enum cudaDriverEntryPointQueryResult *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDART_SYMBOL("cudaGetDriverEntryPoint"));
  auto _ret = func_entry(symbol, funcPtr, flags, driverStatus);
  return _ret;
}

DRYRUN_REGISTE_FUNC(cudaGetDriverEntryPoint)
DRYRUN_REGISTE_FUNC(cudaFuncSetAttribute)
DRYRUN_REGISTE_FUNC(cudaGetLastError)
DRYRUN_REGISTE_FUNC(cudaPeekAtLastError)
DRYRUN_REGISTE_FUNC(cudaPointerGetAttributes)
DRYRUN_REGISTE_FUNC(cudaFuncGetAttributes)
DRYRUN_REGISTE_FUNC(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
DRYRUN_REGISTE_FUNC(cudaDeviceGetStreamPriorityRange)
DRYRUN_REGISTE_FUNC(cudaDeviceGetAttribute)

}  // namespace dryrun
}  // namespace hook
