
#include "header/include/nccl.h"
#include "common/hook.h"
#include "common/api_log.h"
#include "common/black_list.h"
#include "common/macro_common.h"
#include "common/utils.hpp"
#include "nccl/cust/declares.h"

namespace hook {
ncclResult_t ncclAllReduce(const void *, void *, size_t, ncclDataType_t,
                           ncclRedOp_t, ncclComm_t, cudaStream_t) {
  return ncclSuccess;
}

ncclResult_t ncclAllGather(const void *, void *, size_t, ncclDataType_t,
                           ncclComm_t, cudaStream_t) {
  return ncclSuccess;
}

ncclResult_t ncclBcast(void *, size_t, ncclDataType_t, int, ncclComm_t,
                       cudaStream_t) {
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t) { return ncclSuccess; }
ncclResult_t ncclGroupEnd() { return ncclSuccess; }
ncclResult_t ncclGroupStart() { return ncclSuccess; }
ncclResult_t ncclSend(const void *, size_t, ncclDataType_t, int, ncclComm_t,
                      cudaStream_t) {
  return ncclSuccess;
}

ncclResult_t ncclRecv(void *, size_t, ncclDataType_t, int, ncclComm_t,
                      cudaStream_t) {
  return ncclSuccess;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t, ncclResult_t *asyncError) {
  if (asyncError) {
    *asyncError = ncclSuccess;
  }
  return ncclSuccess;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int, ncclUniqueId, int,
                                    ncclConfig_t *) {
  *comm = reinterpret_cast<ncclComm_t>(new int);
  return ncclSuccess;
}

ncclResult_t ncclCommSplit(ncclComm_t, int, int, ncclComm_t *newcomm,
                           ncclConfig_t *) {
  *newcomm = reinterpret_cast<ncclComm_t>(new int);
  return ncclSuccess;
}

ncclResult_t ncclGetVersion(int *version) {
  if (version) {
    *version = 22005;
  }
  return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId) { return ncclSuccess; }

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream) {
  return ncclSuccess;
}

DRYRUN_REGISTE_FUNC(ncclReduceScatter)
DRYRUN_REGISTE_FUNC(ncclCommGetAsyncError)
DRYRUN_REGISTE_FUNC(ncclCommInitRankConfig)
DRYRUN_REGISTE_FUNC(ncclCommSplit)
DRYRUN_REGISTE_FUNC(ncclGetVersion)
DRYRUN_REGISTE_FUNC(ncclGetUniqueId)
DRYRUN_REGISTE_FUNC(ncclAllReduce)
DRYRUN_REGISTE_FUNC(ncclAllGather)
DRYRUN_REGISTE_FUNC(ncclBcast)
DRYRUN_REGISTE_FUNC(ncclCommAbort)
DRYRUN_REGISTE_FUNC(ncclGroupEnd)
DRYRUN_REGISTE_FUNC(ncclGroupStart)
DRYRUN_REGISTE_FUNC(ncclSend)
DRYRUN_REGISTE_FUNC(ncclRecv)

}  // namespace hook
