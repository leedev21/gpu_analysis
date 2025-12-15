#pragma once

#include <sstream>
#include <ostream>
#include "cuda.h"

namespace {
template <>
void _dump_basic<CUdeviceptr>(std::stringstream& s, CUdeviceptr value) {
  if (value) {
    s << demangle(typeid(value).name()) << "="
      << reinterpret_cast<void*>(value);
  }
}

template <>
void _dump_basic<CUpointer_attribute>(std::stringstream& s,
                                      CUpointer_attribute value) {
  s << demangle(typeid(value).name()) << "=" << value;
}

template <>
void _dump_basic<CUfunction_attribute>(std::stringstream& s,
                                       CUfunction_attribute value) {
  s << demangle(typeid(value).name()) << "=" << value;
}

template <>
void _dump_basic<CUdevice_attribute>(std::stringstream& s,
                                     CUdevice_attribute value) {
  s << demangle(typeid(value).name()) << "=" << value;
}

template <>
void _dump_basic<CUdriverProcAddressQueryResult_enum*>(
    std::stringstream& s, CUdriverProcAddressQueryResult_enum* value) {
  s << demangle(typeid(value).name()) << "=";
  if (value) {
    s << *value;
  }
}

template <>
void _dump_basic<unsigned int*>(std::stringstream& s, unsigned int* value) {
  s << demangle(typeid(value).name()) << "=";
  if (value) {
    s << *value;
  }
}

template <>
void _dump_basic<char const*>(std::stringstream& s, char const* value) {
  if (value) {
    s << demangle(typeid(value).name()) << "=" << value;
  } else {
    s << demangle(typeid(value).name()) << "=null";
  }
}
}  // namespace
