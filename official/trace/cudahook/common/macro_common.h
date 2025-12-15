// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 15:40:15 on Sun, May 29, 2022
//
// Description: common macro

#ifndef __CUDA_HOOK_MACRO_COMMON_H__
#define __CUDA_HOOK_MACRO_COMMON_H__

#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string>
#include <sstream>
#include "driver_types.h"

#include "log.h"

#define HOOK_C_API extern "C"
#define HOOK_DECL_EXPORT __attribute__((visibility("default")))

#define HOOK_LIKELY(x) __builtin_expect(!!(x), 1)
#define HOOK_UNLIKELY(x) __builtin_expect(!!(x), 0)

inline char *curr_time() {
  time_t raw_time = time(nullptr);
  struct tm *time_info = localtime(&raw_time);
  static char now_time[64];
  now_time[strftime(now_time, sizeof(now_time), "%Y-%m-%d %H:%M:%S",
                    time_info)] = '\0';

  return now_time;
}

inline int get_pid() {
  static int pid = getpid();

  return pid;
}

inline long int get_tid() {
  thread_local long int tid = syscall(SYS_gettid);

  return tid;
}

#define HOOK_LOG_TAG "CUDA-HOOK"
#define HOOK_LOG_FILE(x) (strrchr(x, '/') ? (strrchr(x, '/') + 1) : x)
#define HLOG(format, ...)                                                 \
  do {                                                                    \
    fprintf(stderr, "[%s %s %d:%ld %s:%d %s] " format "\n", HOOK_LOG_TAG, \
            curr_time(), get_pid(), get_tid(), HOOK_LOG_FILE(__FILE__),   \
            __LINE__, __FUNCTION__, ##__VA_ARGS__);                       \
  } while (0)

#define HOOK_CHECK(x)                                        \
  do {                                                       \
    if (HOOK_UNLIKELY(!(x))) {                               \
      HLOGEX("Check failed %s for %s %s %d\n", #x, __func__, \
             HOOK_LOG_FILE(__FILE__), __LINE__);             \
      exit(EXIT_FAILURE);                                    \
    }                                                        \
  } while (0)

#define HOOK_DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName &) = delete;          \
  void operator=(const TypeName &) = delete;

#include <iostream>
void dump(const struct cudaDeviceProp *, int device = 0);

#ifdef PARAM_LOG
#define LOG IMPL_LOG
#define LOGVAR IMPL_LOGVAR
#define LOGVAR2 IMPL_LOGVAR2
#define LOG_PUINT3 IMPL_LOG_PUINT3
#define LOG_UINT3 IMPL_LOG_UINT3
#define LOG_PINT IMPL_LOG_PINT
#define LOG_cudaPointerAttributes IMPL_LOG_cudaPointerAttributes
#define LOG_cudaFuncAttributes IMPL_LOG_cudaFuncAttributes
#define APPEND IMPL_APPEND
#define LOGPTR IMPL_LOGPTR
#else
#define LOG  //
#define LOGVAR(VAR)
#define LOGVAR2(VAR1, VAR2)
#define LOG_PUINT3(VAR)
#define LOG_UINT3(VAR)
#define LOG_PINT(VAR)
#define LOG_cudaPointerAttributes(VAR)
#define LOG_cudaFuncAttributes(VAR)
#define APPEND(val)
#define LOGPTR(ptr)
#endif

#define NO_LOG  //
#define NO_LOGVAR(VAR)
#define NO_LOGVAR2(VAR1, VAR2)
#define NO_LOG_PUINT3(VAR)
#define NO_LOG_UINT3(VAR)
#define NO_LOG_PINT(VAR)
#define NO_LOG_cudaPointerAttributes(VAR)
#define NO_LOG_cudaFuncAttributes(VAR)
#define NO_APPEND(val)
#define NO_LOGPTR(ptr)

#define IMPL_LOG std::cout << "  line " << __LINE__ << " "

#define IMPL_LOGVAR(VAR)                                              \
  std::cout << "  line " << __LINE__ << " " << #VAR << " val " << VAR \
            << std::endl
#define IMPL_LOGVAR2(VAR1, VAR2)                                             \
  std::cout << "  line " << __LINE__ << " " << #VAR1 << " " << VAR1 << " : " \
            << #VAR2 << " " << VAR2 << std::endl

#define IMPL_LOGPTR(ptr)                                       \
  std::cout << "  line " << __LINE__ << " " << #ptr << " ptr " \
            << reinterpret_cast<const int *>(ptr) << std::endl

#define IMPL_LOG_PUINT3(VAR)                                                  \
  if (VAR) {                                                                  \
    std::cout << "  line " << __LINE__ << " ptr " << #VAR << " x: " << VAR->x \
              << " y: " << VAR->y << " z: " << VAR->z << std::endl;           \
  } else {                                                                    \
    std::cout << "  line " << __LINE__ << " ptr " << #VAR << " nullptr "      \
              << std::endl;                                                   \
  }

#define IMPL_LOG_UINT3(VAR)                                                \
  std::cout << "  line " << __LINE__ << " ptr " << #VAR << " x: " << VAR.x \
            << " y: " << VAR.y << " z: " << VAR.z << std::endl;

#define IMPL_LOG_PINT(VAR)                                               \
  if (VAR) {                                                             \
    std::cout << "  line " << __LINE__ << " ptr " << #VAR << " " << *VAR \
              << std::endl;                                              \
  } else {                                                               \
    std::cout << "  line " << __LINE__ << " ptr " << #VAR << " nullptr " \
              << std::endl;                                              \
  }

#define IMPL_LOG_cudaPointerAttributes(VAR)                                    \
  if (VAR) {                                                                   \
    std::cout << "  line " << __LINE__ << " " << #VAR << "  type "             \
              << VAR->type << "  device " << VAR->device << "  devicePointer " \
              << reinterpret_cast<int *>(VAR->devicePointer)                   \
              << "  hostPointer " << reinterpret_cast<int *>(VAR->hostPointer) \
              << std::endl;                                                    \
  } else {                                                                     \
    std::cout << "  line " << __LINE__ << " ptr " << #VAR << " nullptr "       \
              << std::endl;                                                    \
  }

#define IMPL_LOG_cudaFuncAttributes(VAR)                                 \
  if (VAR) {                                                             \
    LOGVAR(VAR->sharedSizeBytes);                                        \
    LOGVAR(VAR->constSizeBytes);                                         \
    LOGVAR(VAR->localSizeBytes);                                         \
    LOGVAR(VAR->maxThreadsPerBlock);                                     \
    LOGVAR(VAR->numRegs);                                                \
    LOGVAR(VAR->ptxVersion);                                             \
    LOGVAR(VAR->binaryVersion);                                          \
    LOGVAR(VAR->cacheModeCA);                                            \
    LOGVAR(VAR->maxDynamicSharedSizeBytes);                              \
    LOGVAR(VAR->preferredShmemCarveout);                                 \
    LOGVAR(VAR->clusterDimMustBeSet);                                    \
    LOGVAR(VAR->requiredClusterWidth);                                   \
    LOGVAR(VAR->requiredClusterHeight);                                  \
    LOGVAR(VAR->requiredClusterDepth);                                   \
    LOGVAR(VAR->clusterSchedulingPolicyPreference);                      \
    LOGVAR(VAR->nonPortableClusterSizeAllowed);                          \
  } else {                                                               \
    std::cout << "  line " << __LINE__ << " ptr " << #VAR << " nullptr " \
              << std::endl;                                              \
  }

#define IMPL_APPEND(val) str << "\n" << #val << " : " << p->val

#endif  // __CUDA_HOOK_MACRO_COMMON_H__
