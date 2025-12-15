#pragma once

#include <iostream>
#include "common/env_default.hpp"
#include "common/log.h"

#define DECLARE_FUNCS(FUNC)                     \
  void* Get_##FUNC##_Ptr(void* hook = nullptr); \
  void* Get_##FUNC##_NativePtr();

#define NATIVE_REGISTE_FUNC(FUNC)                        \
  struct REGISTE_NATIVE_##FUNC {                         \
    REGISTE_NATIVE_##FUNC() {                            \
      if (0 == VALUE(CH_WORK_MODE))                      \
        Get_##FUNC##_Ptr(reinterpret_cast<void*>(FUNC)); \
    }                                                    \
  } REGISTE_NATIVE_INST##FUNC;

#define DRYRUN_REGISTE_FUNC(FUNC)                        \
  struct REGISTE_DRYRUN_##FUNC {                         \
    REGISTE_DRYRUN_##FUNC() {                            \
      if (1 == VALUE(CH_WORK_MODE))                      \
        Get_##FUNC##_Ptr(reinterpret_cast<void*>(FUNC)); \
    }                                                    \
  } REGISTE_DRYRUN_INST##FUNC;

#define DRYRUN_REGISTE_FUNC2 DRYRUN_REGISTE_FUNC

int getDeviceId();
void setDeviceId(int);

void cudaHookInit();

int getFakeDeviceCount();
