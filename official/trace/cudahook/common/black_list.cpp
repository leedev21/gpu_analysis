#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

#include "common/env_default.hpp"
#include "common/log.h"

namespace {
std::vector<std::string> _getBlackList() {
  auto list = VALUE(CH_API_BLACKLIST);
  list.push_back(std::string("__cudaRegisterFatBinaryEnd"));
  list.push_back(std::string("__cudaRegisterFunction"));

  list.push_back(std::string("__cudaRegisterVar"));  
  list.push_back(std::string("__cudaRegisterFatBinary"));
  list.push_back(std::string("__cudaUnregisterFatBinary"));
  
  // list.push_back(std::string("__cudaPushCallConfiguration"));
  // list.push_back(std::string("__cudaPopCallConfiguration"));

  // list.push_back(std::string("cudaGetDeviceCount"));
  // list.push_back(std::string("cudaGetDevice"));
  // list.push_back(std::string("cudaSetDevice"));
  // list.push_back(std::string("cudaGetDeviceProperties_v2"));
  // list.push_back(std::string("cudaGetLastError"));
  // list.push_back(std::string("cudaStreamIsCapturing"));
  // list.push_back(std::string("cudaMalloc"));
  // list.push_back(std::string("cudaFree"));

  // list.push_back(std::string("cuDevicePrimaryCtxGetState"));
  // list.push_back(std::string("cuGetProcAddress_v2"));
  // list.push_back(std::string("cuGetProcAddress"));
  // list.push_back(std::string("cuPointerGetAttribute"));
  // list.push_back(std::string("cuCtxGetCurrent"));

  // list.push_back(std::string("nvmlInit"));
  // list.push_back(std::string("nvmlDeviceGetCount_v2"));
  return list;
}
}  // namespace

bool isApiInBlackList(const char* api) {
  // static auto ptr = GetHold();
  static auto list = _getBlackList();
  // static auto list = VALUE(CH_API_BLACKLIST);
  if (list.empty()) {
    return false;
  }
  if (std::find(list.begin(), list.end(), std::string("*")) != list.end()) {
    return true;
  }
  return std::find(list.begin(), list.end(), std::string(api)) != list.end();
}
