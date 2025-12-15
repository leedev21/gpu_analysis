#pragma once

#include "cuda_runtime_api.h"

namespace hook {
const char* getErrorString(cudaError_t);
}  // namespace hook
