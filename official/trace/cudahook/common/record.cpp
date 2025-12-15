#include <cxxabi.h>
#include <memory>
#include "common/record.h"
#include "common/env_default.hpp"
#include "common/log.h"

std::string demangle(const char* name) {
  int status = -1;

  // Use __cxa_demangle to demangle the symbol name
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};

  return (status == 0) ? res.get() : name;
}

void PtrNameMap::Registe(const void* ptr, const char* fun) {
  std::lock_guard<std::mutex> lg(mtx_);
  mpas_[ptr] = std::string(demangle(fun));
}

std::string PtrNameMap::Query(const void* ptr) {
  std::lock_guard<std::mutex> lg(mtx_);
  do {
    auto it = mpas_.find(ptr);
    if (it == mpas_.end()) {
      break;
    }
    return it->second;
  } while (0);
  return nullptr;
}

PtrNameMap* GetKernelNameStore() {
  static PtrNameMap inst;
  return &inst;
}

void DumpKernelName(const void* func, const char*) {
  if (1 == VALUE(CH_DBG_DUMP_KERNEL_NAME)) {
    auto func_name = GetKernelNameStore()->Query(func);
    if (!func_name.empty()) {
      HLOGEX("kernel name %s\n", func_name.c_str());
    }
  }
}
