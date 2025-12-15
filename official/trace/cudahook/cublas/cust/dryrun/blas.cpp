#define CUBLASAPI

// auto generate 525 apis

#include <cstdint>

#include "cublas_api.h"
#include "common/hook.h"
#include "common/api_log.h"
#include "common/black_list.h"
#include "common/utils.hpp"
#include "cublas/cust/declares.h"

namespace hook {
struct MycublasHandle_t {
  cublasMath_t mode_ = CUBLAS_DEFAULT_MATH;
  cudaStream_t stream_ = nullptr;
};

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
  auto ret = new MycublasHandle_t;
  *handle = reinterpret_cast<cublasHandle_t>(ret);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) {
  auto h = reinterpret_cast<MycublasHandle_t*>(handle);
  *mode = h->mode_;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
  auto h = reinterpret_cast<MycublasHandle_t*>(handle);
  h->mode_ = mode;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream_v2(cublasHandle_t handle,
                                  cudaStream_t* streamId) {
  auto h = reinterpret_cast<MycublasHandle_t*>(handle);
  *streamId = h->stream_;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype,
    int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb,
    long long int strideB, const void* beta, void* C, cudaDataType Ctype,
    int ldc, long long int strideC, int batchCount,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  // FAKE_LOG(__func__);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void* alpha, const void* A,
                            cudaDataType Atype, int lda, const void* B,
                            cudaDataType Btype, int ldb, const void* beta,
                            void* C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t computeType,
                            cublasGemmAlgo_t algo) {
  // FAKE_LOG(__func__);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t handle, void* workspace,
                                     size_t workspaceSizeInBytes) {
  // FAKE_LOG(__func__);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream_v2(cublasHandle_t handle,
                                  cudaStream_t streamId) {
  auto h = reinterpret_cast<MycublasHandle_t*>(handle);
  h->stream_ = streamId;
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t, cublasOperation_t,
                                         cublasOperation_t, int, int, int,
                                         const float*, const float*, int,
                                         long long int, const float*, int,
                                         long long int, const float*, float*,
                                         int, long long int, int) {
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k,
                              const float* alpha, const float* A, int lda,
                              const float* B, int ldb, const float* beta,
                              float* C, int ldc) {
  return CUBLAS_STATUS_SUCCESS;
}

DRYRUN_REGISTE_FUNC(cublasCreate_v2)
DRYRUN_REGISTE_FUNC(cublasGetMathMode)
DRYRUN_REGISTE_FUNC(cublasSetMathMode)
DRYRUN_REGISTE_FUNC(cublasGetStream_v2)
DRYRUN_REGISTE_FUNC(cublasGemmStridedBatchedEx)
DRYRUN_REGISTE_FUNC(cublasGemmEx)
DRYRUN_REGISTE_FUNC(cublasSetWorkspace_v2)
DRYRUN_REGISTE_FUNC(cublasSetStream_v2)
DRYRUN_REGISTE_FUNC(cublasSgemmStridedBatched)
DRYRUN_REGISTE_FUNC(cublasSgemm_v2)

}  // namespace hook
