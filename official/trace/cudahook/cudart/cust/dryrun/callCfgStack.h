#pragma once

#include "cuda_runtime_api.h"

#include <cstdint>
#include <stack>
// #include <map>
#include <mutex>

namespace hook {
struct CallCfg {
  dim3 grid;
  dim3 block;
  size_t mem;
  cudaStream_t stream;
};

class CallCfgStack {
 protected:
  std::mutex mtx_;
  std::stack<CallCfg> cfg_;

 public:
  CallCfgStack() = default;
  ~CallCfgStack() = default;

 public:
  bool Push(dim3, dim3, size_t, void*);
  bool Pop(dim3*, dim3*, size_t*, void*);

  static CallCfgStack* getInstance();
};
}  // namespace hook