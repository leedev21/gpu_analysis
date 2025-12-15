#pragma once

#include <map>
#include <string>
#include <mutex>

class PtrNameMap {
 private:
  std::map<const void*, std::string> mpas_;
  mutable std::mutex mtx_;

 public:
  PtrNameMap() = default;
  ~PtrNameMap() = default;

  void Registe(const void*, const char*);
  std::string Query(const void*);
};

PtrNameMap* GetKernelNameStore();

void DumpKernelName(const void*, const char* api = nullptr);
