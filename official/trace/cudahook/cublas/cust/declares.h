#pragma once

#include "common/utils.hpp"

namespace hook {
// cublas
DECLARE_FUNCS(cublasCreate_v2);
DECLARE_FUNCS(cublasGetMathMode);
DECLARE_FUNCS(cublasSetMathMode);
DECLARE_FUNCS(cublasGetStream_v2);
DECLARE_FUNCS(cublasGemmStridedBatchedEx);
DECLARE_FUNCS(cublasGemmEx);
DECLARE_FUNCS(cublasSetWorkspace_v2);
DECLARE_FUNCS(cublasSetStream_v2);
DECLARE_FUNCS(cublasSgemmStridedBatched);
DECLARE_FUNCS(cublasSgemm_v2);

// cublasLt
DECLARE_FUNCS(cublasLtMatmul);
DECLARE_FUNCS(cublasLtMatmulPreferenceSetAttribute);
DECLARE_FUNCS(cublasLtCreate);
DECLARE_FUNCS(cublasLtMatrixLayoutCreate);
DECLARE_FUNCS(cublasLtMatrixLayoutDestroy);
DECLARE_FUNCS(cublasLtMatmulPreferenceCreate);
DECLARE_FUNCS(cublasLtMatmulPreferenceDestroy);
DECLARE_FUNCS(cublasLtMatmulDescCreate);
DECLARE_FUNCS(cublasLtMatmulDescDestroy);
DECLARE_FUNCS(cublasLtMatmulDescSetAttribute);
DECLARE_FUNCS(cublasLtMatmulAlgoGetHeuristic);

}  // namespace hook
