#pragma once
#include <sstream>
#include <ostream>

#include "cudnn_graph.h"

#include "common/hook.h"
#include "common/api_log.h"
#include "dnn/api_log.h"

namespace {
template <>
void _dump_basic<cudnnBackendAttributeType_t>(std::stringstream& s,
                                              cudnnBackendAttributeType_t v) {
  s << demangle(typeid(v).name()) << " = " << v;
}

template <>
void _dump_basic<cudnnBackendAttributeName_t>(std::stringstream& s,
                                              cudnnBackendAttributeName_t v) {
  s << demangle(typeid(v).name()) << " = " << v;
}

template <>
void _dump_basic<cudnnBackendDescriptorType_t>(std::stringstream& s,
                                               cudnnBackendDescriptorType_t v) {
  s << demangle(typeid(v).name()) << " = " << v;
}

template <>
void _dump_basic<cudnnHandle_t*>(std::stringstream& s, cudnnHandle_t* v) {
  if (v) {
    s << demangle(typeid(v).name()) << " = " << reinterpret_cast<void*>(*v);
  }
}

template <>
void _dump_basic<int64_t*>(std::stringstream& s, int64_t* v) {
  if (v) {
    s << demangle(typeid(v).name()) << " = " << *v;
  }
}

template <>
void _dump_basic<int64_t>(std::stringstream& s, int64_t v) {
  s << demangle(typeid(v).name()) << " = " << v;
}

}  // namespace
