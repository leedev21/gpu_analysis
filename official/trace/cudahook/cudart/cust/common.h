
#include "cuda_runtime_api.h"

namespace hook {
cudaError_t hookCudaGetDriverEntryPoint(const char *, void **,
                                        unsigned long long,
                                        enum cudaDriverEntryPointQueryResult *);
}
