#pragma once

#include "common/utils.hpp"

namespace hook {
DECLARE_FUNCS(ncclCommGetAsyncError)
DECLARE_FUNCS(ncclCommInitRankConfig)
DECLARE_FUNCS(ncclCommSplit)
DECLARE_FUNCS(ncclGetVersion)
DECLARE_FUNCS(ncclGetUniqueId)
DECLARE_FUNCS(ncclAllReduce)
DECLARE_FUNCS(ncclAllGather)
DECLARE_FUNCS(ncclBcast)
DECLARE_FUNCS(ncclCommAbort)
DECLARE_FUNCS(ncclGroupEnd)
DECLARE_FUNCS(ncclGroupStart)
DECLARE_FUNCS(ncclSend)
DECLARE_FUNCS(ncclRecv)
DECLARE_FUNCS(ncclReduceScatter)

}  // namespace hook
