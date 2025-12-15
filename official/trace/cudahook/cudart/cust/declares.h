#pragma once

#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_API_PER_THREAD_DEFAULT_STREAM
#define __CUDA_API_VERSION_INTERNAL

// auto generate 2 apis

#include "common/utils.hpp"

namespace hook {
DECLARE_FUNCS(__cudaPushCallConfiguration)
DECLARE_FUNCS(__cudaPopCallConfiguration)
DECLARE_FUNCS(__cudaRegisterFunction)
DECLARE_FUNCS(__cudaRegisterVar)

DECLARE_FUNCS(cudaStreamCreate)
DECLARE_FUNCS(cudaStreamCreateWithFlags)
DECLARE_FUNCS(cudaStreamDestroy)

DECLARE_FUNCS(cudaStreamCreateWithPriority)
DECLARE_FUNCS(cudaStreamSynchronize)
DECLARE_FUNCS(cudaStreamIsCapturing)
DECLARE_FUNCS(cudaDeviceGetStreamPriorityRange)
DECLARE_FUNCS(cudaStreamWaitEvent)
DECLARE_FUNCS(cudaEventSynchronize)

DECLARE_FUNCS(cudaEventCreate)
DECLARE_FUNCS(cudaEventCreateWithFlags)
DECLARE_FUNCS(cudaEventDestroy)
DECLARE_FUNCS(cudaEventQuery)
DECLARE_FUNCS(cudaEventRecord)
DECLARE_FUNCS(cudaEventElapsedTime)

DECLARE_FUNCS(cudaLaunchCooperativeKernel)
DECLARE_FUNCS(cudaLaunchKernel)
DECLARE_FUNCS(cudaLaunchKernelExC)

DECLARE_FUNCS(cudaMalloc)
DECLARE_FUNCS(cudaFree)
DECLARE_FUNCS(cudaHostAlloc)
DECLARE_FUNCS(cudaMemsetAsync)
DECLARE_FUNCS(cudaMemcpyAsync)
DECLARE_FUNCS(cudaMemcpy)

DECLARE_FUNCS(cudaGetLastError)
DECLARE_FUNCS(cudaPeekAtLastError)

DECLARE_FUNCS(cudaDeviceGetAttribute)
DECLARE_FUNCS(cudaGetDeviceProperties_v2)
DECLARE_FUNCS(cudaGetDeviceCount)
DECLARE_FUNCS(cudaSetDevice)
DECLARE_FUNCS(cudaGetDevice);

DECLARE_FUNCS(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)
DECLARE_FUNCS(cudaDeviceSynchronize)

DECLARE_FUNCS(cudaFuncGetAttributes)
DECLARE_FUNCS(cudaFuncSetAttribute)
DECLARE_FUNCS(cudaPointerGetAttributes)

DECLARE_FUNCS(__cudaRegisterFatBinary)
DECLARE_FUNCS(__cudaRegisterFatBinaryEnd)
DECLARE_FUNCS(__cudaUnregisterFatBinary)

DECLARE_FUNCS(cudaGetDriverEntryPoint)

}  // namespace hook
