#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_API_PER_THREAD_DEFAULT_STREAM
#define __CUDA_API_VERSION_INTERNAL

#include <vector>
#include <iostream>
#include <stdio.h>

#include "cuda_runtime_api.h"
#include "common/utils.hpp"
#include "common/hook.h"
#include "common/env_default.hpp"

// #define PROP_FILE "/home/fred.zhao/devProp.3060"
//  #define PROP_FILE "/home/fred.zhao/devProp.v100"

namespace {
void _readDeviceProp(cudaDeviceProp* p) {
  using func_ptr = cudaError_t (*)(struct cudaDeviceProp*, int);
  static auto fn = reinterpret_cast<func_ptr>(
      HOOK_CUDART_SYMBOL("cudaGetDeviceProperties_v2"));
  HOOK_CHECK(fn);
  auto ret = fn(p, 0);

  // {
  //   FILE* f = fopen(PROP_FILE, "wb");
  //   if (f) {
  //     fwrite(p, sizeof(*p), 1, f);
  //     fclose(f);
  //   }
  // }
}

void _readDevicePropFromFile(cudaDeviceProp* p) {
  static const char* device_s[] = {
      "/home/fred.zhao/devProp.3060",  //
      "/home/fred.zhao/devProp.v100",  //
      "/home/fred.zhao/devProp.a100",  //
      "/home/fred.zhao/devProp.h100",  //
  };
  auto idx = VALUE(CH_VIRT_DEVICE_ID);
  FILE* f = fopen(device_s[idx], "rb");
  if (f) {
    fread(p, sizeof(*p), 1, f);
    fclose(f);
    std::cout << __func__ << " " << p->name << "\n";
  }
}

// todo from file
void createDeviceProp(cudaDeviceProp* p) {
  // _readDevicePropFromFile(p);
  _readDeviceProp(p);
}

class DevicePropManager {
  std::vector<cudaDeviceProp> props_;

 public:
  explicit DevicePropManager(unsigned int);
  ~DevicePropManager() = default;
  void Query(cudaDeviceProp* dst, unsigned int id) {
    memcpy(dst, props_.data() + id, sizeof(cudaDeviceProp));
  }
  static DevicePropManager* getInstance();
};

DevicePropManager::DevicePropManager(unsigned int dev_count) {
  props_.resize(dev_count);
  createDeviceProp(props_.data());
  for (size_t i = 1; i < props_.size(); i++) {
    memcpy(props_.data() + i, props_.data(), sizeof(cudaDeviceProp));
  }
}

DevicePropManager* DevicePropManager::getInstance() {
  static DevicePropManager inst(VALUE(CH_VIRT_DEVICE_COUNT));
  return &inst;
}
}  // namespace

namespace hook {
bool QueryDeviceProp(cudaDeviceProp* dst, unsigned int id) {
  DevicePropManager::getInstance()->Query(dst, id);
  return true;
}
}  // namespace hook
