#pragma once

#include "common/utils.hpp"

namespace hook {
DECLARE_FUNCS(cuPointerGetAttribute)
DECLARE_FUNCS(cuLaunchKernel)
DECLARE_FUNCS(cuLaunchKernel_ptsz)
DECLARE_FUNCS(cuDevicePrimaryCtxGetState)
DECLARE_FUNCS(cuDeviceGet)
DECLARE_FUNCS(cuDeviceGetAttribute)

DECLARE_FUNCS(cuFuncGetAttribute)
DECLARE_FUNCS(cuCtxGetCurrent)
DECLARE_FUNCS(cuCtxSetCurrent)
DECLARE_FUNCS(cuDevicePrimaryCtxRetain)
DECLARE_FUNCS(cuModuleGetFunction)
DECLARE_FUNCS(cuModuleLoadData)
DECLARE_FUNCS(cuGetProcAddress_v2)
DECLARE_FUNCS(cuGetProcAddress)
DECLARE_FUNCS(cuInit)

DECLARE_FUNCS(cuMemMap)
DECLARE_FUNCS(cuMemSetAccess)
DECLARE_FUNCS(cuMemCreate)

DECLARE_FUNCS(cuFuncSetAttribute)
DECLARE_FUNCS(cuOccupancyAvailableDynamicSMemPerBlock)
DECLARE_FUNCS(cuOccupancyMaxActiveBlocksPerMultiprocessor)

DECLARE_FUNCS(cuCtxGetDevice)
DECLARE_FUNCS(cuDriverGetVersion)
DECLARE_FUNCS(cuKernelGetFunction)
DECLARE_FUNCS(cuKernelSetAttribute)
DECLARE_FUNCS(cuLibraryGetKernel)
DECLARE_FUNCS(cuLibraryLoadData)
DECLARE_FUNCS(cuLibraryUnload)
DECLARE_FUNCS(cuOccupancyMaxActiveClusters)
DECLARE_FUNCS(cuMemAddressReserve)
DECLARE_FUNCS(cuModuleLoadDataEx)

}  // namespace hook
