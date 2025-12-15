#define __CUDA_API_PER_THREAD_DEFAULT_STREAM
#define __CUDA_API_VERSION_INTERNAL

// auto generate 588 apis

#include "cuda.h"
#include "common/hook.h"
#include "common/utils.hpp"
#include "common/macro_common.h"
#include "cuda/cust/declares.h"
#include "cuda/error_check.h"

namespace hook {
namespace dryrun {
using DryrunCUmemGenericAllocationHandle = CUmemGenericAllocationHandle;
CUresult cuLaunchKernel(CUfunction, unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int, unsigned int,
                        CUstream, void **, void **) {
  return CUDA_SUCCESS;
}
CUresult cuLaunchKernel_ptsz(CUfunction, unsigned int, unsigned int,
                             unsigned int, unsigned int, unsigned int,
                             unsigned int, unsigned int, CUstream, void **,
                             void **) {
  return CUDA_SUCCESS;
}

CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute,
                               CUdeviceptr ptr) {
  if (CU_POINTER_ATTRIBUTE_DEVICE_POINTER == attribute) {
    *reinterpret_cast<CUdeviceptr *>(data) = ptr;
  }
  return CUDA_SUCCESS;
}

CUresult cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle,
                  unsigned long long) {
  return CUDA_SUCCESS;
}

CUresult cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t) {
  return CUDA_SUCCESS;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t,
                     const CUmemAllocationProp *, unsigned long long) {
  *handle = reinterpret_cast<CUmemGenericAllocationHandle>(
      new DryrunCUmemGenericAllocationHandle);
  return CUDA_SUCCESS;
}

DRYRUN_REGISTE_FUNC2(cuLaunchKernel)
DRYRUN_REGISTE_FUNC2(cuLaunchKernel_ptsz)
DRYRUN_REGISTE_FUNC2(cuPointerGetAttribute)
DRYRUN_REGISTE_FUNC2(cuMemMap)
DRYRUN_REGISTE_FUNC2(cuMemSetAccess)
DRYRUN_REGISTE_FUNC2(cuMemCreate)

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags,
                                    int *active) {
  using func_ptr = CUresult (*)(CUdevice, unsigned int *, int *);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDA_SYMBOL("cuDevicePrimaryCtxGetState"));
  HOOK_CHECK(func_entry);
  auto ret = func_entry(0, flags, active);
  return ret;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  using func_ptr = CUresult (*)(CUdevice *, int);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGet"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(device, 0);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib,
                              CUdevice dev) {
  using func_ptr = CUresult (*)(int *, CUdevice_attribute, CUdevice);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDeviceGetAttribute"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(pi, attrib, 0);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
  using func_ptr = CUresult (*)(CUcontext *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxGetCurrent"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(pctx);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  using func_ptr = CUresult (*)(CUcontext);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxSetCurrent"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(ctx);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
  using func_ptr = CUresult (*)(CUcontext *, CUdevice);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDevicePrimaryCtxRetain"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(pctx, dev);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuFuncGetAttribute(int *pi, CUfunction_attribute attrib,
                            CUfunction hfunc) {
  using func_ptr = CUresult (*)(int *, CUfunction_attribute, CUfunction);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuFuncGetAttribute"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(pi, attrib, hfunc);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                             cuuint64_t flags,
                             CUdriverProcAddressQueryResult *symbolStatus) {
  using func_ptr = CUresult (*)(const char *, void **, int, cuuint64_t,
                                CUdriverProcAddressQueryResult *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGetProcAddress_v2"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(symbol, pfn, cudaVersion, flags, symbolStatus);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                             const char *name) {
  using func_ptr = CUresult (*)(CUfunction *, CUmodule, const char *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuModuleGetFunction"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(hfunc, hmod, name);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
  using func_ptr = CUresult (*)(CUmodule *, const void *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuModuleLoadData"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(module, image);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuFuncSetAttribute(CUfunction hfunc, CUfunction_attribute attrib,
                            int value) {
  using func_ptr = CUresult (*)(CUfunction, CUfunction_attribute, int);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuFuncSetAttribute"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(hfunc, attrib, value);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuOccupancyAvailableDynamicSMemPerBlock(size_t *dynamicSmemSize,
                                                 CUfunction func, int numBlocks,
                                                 int blockSize) {
  using func_ptr = CUresult (*)(size_t *, CUfunction, int, int);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDA_SYMBOL("cuOccupancyAvailableDynamicSMemPerBlock"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(dynamicSmemSize, func, numBlocks, blockSize);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                     CUfunction func,
                                                     int blockSize,
                                                     size_t dynamicSMemSize) {
  using func_ptr = CUresult (*)(int *, CUfunction, int, size_t);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDA_SYMBOL("cuOccupancyMaxActiveBlocksPerMultiprocessor"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(numBlocks, func, blockSize, dynamicSMemSize);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuCtxGetDevice(CUdevice *device) {
  using func_ptr = CUresult (*)(CUdevice *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuCtxGetDevice"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(device);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuDriverGetVersion(int *driverVersion) {
  using func_ptr = CUresult (*)(int *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuDriverGetVersion"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(driverVersion);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                          cuuint64_t flags) {
  using func_ptr = CUresult (*)(const char *, void **, int, cuuint64_t);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuGetProcAddress"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(symbol, pfn, cudaVersion, flags);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuInit(unsigned int Flags) {
  using func_ptr = CUresult (*)(unsigned int);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuInit"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(Flags);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuKernelGetFunction(CUfunction *pFunc, CUkernel kernel) {
  using func_ptr = CUresult (*)(CUfunction *, CUkernel);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuKernelGetFunction"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(pFunc, kernel);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val,
                              CUkernel kernel, CUdevice dev) {
  using func_ptr = CUresult (*)(CUfunction_attribute, int, CUkernel, CUdevice);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuKernelSetAttribute"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(attrib, val, kernel, dev);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuLibraryGetKernel(CUkernel *pKernel, CUlibrary library,
                            const char *name) {
  using func_ptr = CUresult (*)(CUkernel *, CUlibrary, const char *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLibraryGetKernel"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(pKernel, library, name);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuLibraryLoadData(CUlibrary *library, const void *code,
                           CUjit_option *jitOptions, void **jitOptionsValues,
                           unsigned int numJitOptions,
                           CUlibraryOption *libraryOptions,
                           void **libraryOptionValues,
                           unsigned int numLibraryOptions) {
  using func_ptr =
      CUresult (*)(CUlibrary *, const void *, CUjit_option *, void **,
                   unsigned int, CUlibraryOption *, void **, unsigned int);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLibraryLoadData"));
  HOOK_CHECK(func_entry);
  auto _ret =
      func_entry(library, code, jitOptions, jitOptionsValues, numJitOptions,
                 libraryOptions, libraryOptionValues, numLibraryOptions);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuLibraryUnload(CUlibrary library) {
  using func_ptr = CUresult (*)(CUlibrary);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuLibraryUnload"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(library);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuOccupancyMaxActiveClusters(int *numClusters, CUfunction func,
                                      const CUlaunchConfig *config) {
  using func_ptr = CUresult (*)(int *, CUfunction, const CUlaunchConfig *);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDA_SYMBOL("cuOccupancyMaxActiveClusters"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(numClusters, func, config);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment,
                             CUdeviceptr addr, unsigned long long flags) {
  using func_ptr = CUresult (*)(CUdeviceptr *, size_t, size_t, CUdeviceptr,
                                unsigned long long);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDA_SYMBOL("cuMemAddressReserve"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(ptr, size, alignment, addr, flags);
  CHECK_RETURN(_ret);
  return _ret;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, CUjit_option *options,
                            void **optionValues) {
  using func_ptr = CUresult (*)(CUmodule *, const void *, unsigned int,
                                CUjit_option *, void **);
  static auto func_entry =
      reinterpret_cast<func_ptr>(hook::Get_cuModuleLoadDataEx_Ptr());
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(module, image, numOptions, options, optionValues);
  return _ret;
}

DRYRUN_REGISTE_FUNC(cuModuleLoadDataEx)
DRYRUN_REGISTE_FUNC(cuDevicePrimaryCtxGetState)
DRYRUN_REGISTE_FUNC(cuDeviceGet)
DRYRUN_REGISTE_FUNC(cuDeviceGetAttribute)
DRYRUN_REGISTE_FUNC(cuCtxSetCurrent)
DRYRUN_REGISTE_FUNC(cuCtxGetCurrent)
DRYRUN_REGISTE_FUNC(cuDevicePrimaryCtxRetain)
DRYRUN_REGISTE_FUNC(cuFuncGetAttribute)
DRYRUN_REGISTE_FUNC(cuGetProcAddress_v2)
DRYRUN_REGISTE_FUNC(cuModuleGetFunction)
DRYRUN_REGISTE_FUNC(cuModuleLoadData)
DRYRUN_REGISTE_FUNC(cuFuncSetAttribute)
DRYRUN_REGISTE_FUNC(cuOccupancyAvailableDynamicSMemPerBlock)
DRYRUN_REGISTE_FUNC(cuOccupancyMaxActiveBlocksPerMultiprocessor)
DRYRUN_REGISTE_FUNC(cuMemAddressReserve)
DRYRUN_REGISTE_FUNC(cuCtxGetDevice)
DRYRUN_REGISTE_FUNC(cuDriverGetVersion)
DRYRUN_REGISTE_FUNC(cuGetProcAddress)
DRYRUN_REGISTE_FUNC(cuInit)
DRYRUN_REGISTE_FUNC(cuKernelGetFunction)
DRYRUN_REGISTE_FUNC(cuKernelSetAttribute)
DRYRUN_REGISTE_FUNC(cuLibraryGetKernel)
DRYRUN_REGISTE_FUNC(cuLibraryLoadData)
DRYRUN_REGISTE_FUNC(cuLibraryUnload)
DRYRUN_REGISTE_FUNC(cuOccupancyMaxActiveClusters)
}  // namespace dryrun
}  // namespace hook
