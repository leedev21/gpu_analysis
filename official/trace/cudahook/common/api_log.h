#pragma once

#include <sstream>
#include <typeinfo>
#include "common/log.h"
#include <cxxabi.h>
#include <memory>

inline std::string demangle(const char* name) {
  int status = -1;

  // Use __cxa_demangle to demangle the symbol name
  std::unique_ptr<char, void (*)(void*)> res{
      abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free};

  return (status == 0) ? res.get() : name;
}

namespace {
template <typename T>
void _dump_basic(std::stringstream& s, T value) {
  s << demangle(typeid(value).name());  // << "=" << value;
}

// dump some type
template <>
void _dump_basic<int>(std::stringstream& s, int value) {
  s << demangle(typeid(value).name()) << "=" << value;
}

template <>
void _dump_basic<void*>(std::stringstream& s, void* value) {
  s << demangle(typeid(value).name()) << "=" << value;
}
template <>
void _dump_basic<const void*>(std::stringstream& s, const void* value) {
  s << demangle(typeid(value).name()) << "=" << value;
}
template <>
void _dump_basic<void**>(std::stringstream& s, void** value) {
  if (value) {
    s << demangle(typeid(value).name()) << "=" << *value;
  }
}

template <>
void _dump_basic<unsigned long>(std::stringstream& s, unsigned long value) {
  s << demangle(typeid(value).name()) << "=" << value;
}

template <>
void _dump_basic<unsigned int>(std::stringstream& s, unsigned int value) {
  s << demangle(typeid(value).name()) << "=" << value;
}

template <>
void _dump_basic<int*>(std::stringstream& s, int* value) {
  if (value) {
    s << demangle(typeid(value).name()) << "=" << *value;
  }
}

///////////////////////////////////////////////////////////////////////////
template <typename T>
void _dump_st(std::stringstream& s, T& value) {
  s << demangle(typeid(value).name());  // << "=" << value;
}

template <typename T>
void _dump_one_param(std::stringstream& s, T& value) {
  if (std::is_class<T>::value) {
    _dump_st(s, value);
  } else {
    _dump_basic(s, value);
  }
}
}  // namespace

class ParamDumper {
 private:
  std::stringstream ss_;

 public:
  ParamDumper(const char* func) { ss_ << func << " ("; }
  ~ParamDumper() {
    ss_ << ")";
    HLOGEX("%s", ss_.str().c_str());
  }

  void Dump() {}

  template <typename T>
  void Dump(T value) {
    _dump_one_param(ss_, value);
  }

  template <typename T, typename... Args>
  void Dump(T first, Args... rest) {
    _dump_one_param(ss_, first);
    ss_ << ", ";
    Dump(rest...);
  }
};
