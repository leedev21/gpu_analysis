#pragma once

#include "common/utils.hpp"

namespace hook {
DECLARE_FUNCS(cudnnBackendCreateDescriptor)
DECLARE_FUNCS(cudnnBackendDestroyDescriptor)
DECLARE_FUNCS(cudnnBackendExecute)
DECLARE_FUNCS(cudnnBackendFinalize)
DECLARE_FUNCS(cudnnBackendGetAttribute)
DECLARE_FUNCS(cudnnBackendSetAttribute)
DECLARE_FUNCS(cudnnCreate)
DECLARE_FUNCS(cudnnGetErrorString)
DECLARE_FUNCS(cudnnGetVersion)
DECLARE_FUNCS(cudnnSetStream)
DECLARE_FUNCS(cudnnGetStream)
DECLARE_FUNCS(cudnnDestroy)

}
