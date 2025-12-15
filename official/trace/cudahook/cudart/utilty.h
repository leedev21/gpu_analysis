#pragma once
#include <sstream>
#include <ostream>

#define __CUDA_API_VERSION_INTERNAL
#define __CUDA_API_PER_THREAD_DEFAULT_STREAM
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__

#include "cuda_runtime_api.h"

inline std::ostream& operator<<(std::ostream& o, const struct cudaExtent&) {
  return o;
}

inline std::ostream& operator<<(std::ostream& o,
                                const struct cudaMemLocation&) {
  return o;
}

inline std::ostream& operator<<(std::ostream& o, const struct dim3&) {
  return o;
}

inline std::ostream& operator<<(std::ostream& o,
                                const struct cudaIpcMemHandle_st&) {
  return o;
}

inline std::ostream& operator<<(std::ostream& o,
                                const struct cudaIpcEventHandle_st&) {
  return o;
}

inline std::ostream& operator<<(std::ostream& o, const struct cudaPitchedPtr&) {
  return o;
}

namespace {

template <>
void _dump_st<dim3>(std::stringstream& s, dim3& v) {
  s << demangle(typeid(v).name()) << "={" << v.x << ":" << v.y << ":" << v.z
    << "}";
}

template <>
void _dump_basic<cudaStream_t>(std::stringstream& s, cudaStream_t v) {
  s << demangle(typeid(v).name()) << "=" << reinterpret_cast<void*>(v);
}

template <>
void _dump_basic<cudaStream_t*>(std::stringstream& s, cudaStream_t* v) {
  s << demangle(typeid(v).name()) << "=" << reinterpret_cast<cudaStream_t>(*v);
}

template <>
void _dump_basic<cudaPointerAttributes*>(std::stringstream& s,
                                         cudaPointerAttributes* value) {
  s << demangle(typeid(value).name()) << "=" << value->type << " dev "
    << value->device;
}

template <>
void _dump_basic<cudaStreamCaptureStatus*>(std::stringstream& s,
                                           cudaStreamCaptureStatus* v) {
  if (v) {
    s << demangle(typeid(v).name()) << "=" << *v;
  }
}

template <>
void _dump_basic<cudaDeviceProp*>(std::stringstream& s, cudaDeviceProp* v) {
  int64_t* pu = reinterpret_cast<int64_t*>(v->uuid.bytes);
  int64_t* pl = reinterpret_cast<int64_t*>(v->luid);
  // s << demangle(typeid(v).name()) << "=" << v->name << " " << str_uuid << " "
  // << str_luid;
  s << demangle(typeid(v).name()) << "=" << v->name << " " << std::hex << *pl
    << " 0x" << pu[0] << pu[1];
}

template <>
void _dump_basic<char*>(std::stringstream& s, char* v) {
  if (v) {
    s << demangle(typeid(v).name()) << "=" << reinterpret_cast<void*>(v);
  }
}

template <>
void _dump_basic<char const*>(std::stringstream& s, char const* v) {
  if (v) {
    s << demangle(typeid(v).name()) << "=" << reinterpret_cast<void const*>(v);
  }
}

template <>
void _dump_basic<uint3*>(std::stringstream& s, uint3* v) {
  s << demangle(typeid(v).name());
  if (v) {
    s << "=[" << v->x << ":" << v->y << ":" << v->z << "] ";
  } else {
    s << "=null";
  }
}

template <>
void _dump_basic<dim3*>(std::stringstream& s, dim3* v) {
  s << demangle(typeid(v).name());
  if (v) {
    s << "=[" << v->x << ":" << v->y << ":" << v->z << "] ";
  } else {
    s << "=null";
  }
}

template <>
void _dump_basic<cudaFuncAttributes*>(std::stringstream& s,
                                      cudaFuncAttributes* v) {
  s << demangle(typeid(v).name());
  if (v) {
    s << reinterpret_cast<void*>(v);
  } else {
    s << "=null";
  }
}

template <>
void _dump_basic<cudaFuncAttribute>(std::stringstream& s, cudaFuncAttribute v) {
  s << demangle(typeid(v).name()) << "=" << v;
}

template <>
void _dump_basic<cudaDeviceAttr>(std::stringstream& s, cudaDeviceAttr v) {
  s << demangle(typeid(v).name()) << "= " << v;
}

}  // namespace
