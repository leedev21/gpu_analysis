#include "cudart/cust/dryrun/callCfgStack.h"
// #include "common/macro_common.h"
#include "common/error_check_base.h"


namespace hook {

bool CallCfgStack::Push(dim3 g, dim3 b, size_t m, void* s) {
  std::lock_guard<std::mutex> lock(mtx_);
  cfg_.push({g, b, m, reinterpret_cast<cudaStream_t>(s)});
  return true;
}

bool CallCfgStack::Pop(dim3* g, dim3* b, size_t* m, void* s) {
  std::lock_guard<std::mutex> lock(mtx_);
  do {
    CHECK_RETURN(false == cfg_.empty());
    auto cfg = cfg_.top();
    *g = cfg.grid;
    *b = cfg.block;
    *m = cfg.mem;
    *(reinterpret_cast<cudaStream_t*>(s)) = cfg.stream;
    cfg_.pop();
    return true;
  } while (0);
  return false;
}

thread_local CallCfgStack inst;
CallCfgStack* CallCfgStack::getInstance() {
  // static CallCfgStack inst;
  return &inst;
}
}  // namespace hook
