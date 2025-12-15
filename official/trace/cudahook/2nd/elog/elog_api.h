#pragma once

#if defined(_MSC_VER)
#include <windows.h>
#else
#include <unistd.h>
#include <cxxabi.h>
#endif

#include "elog/elog.h"

#include <typeinfo>
#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <ctime>
#include <chrono>

namespace elog {

unsigned int ELOG_EXPORT eapi_get_processid();
std::chrono::system_clock::time_point& ELOG_EXPORT eapi_time_begin();

template<typename T>
class eapi_typename {
public:
  friend std::ostream& operator<<(std::ostream& os, const eapi_typename& rhs) {
    char buf[1024] = {0};
    size_t len = sizeof(buf);
#if defined(_MSC_VER)
    strncpy(buf, typeid(T).name(), len);
    for (int i = 0; i < strlen(buf); i++)
      if (buf[i] == ' ') os << &buf[i + 1];
    os << buf;
#else
    os << abi::__cxa_demangle(typeid(T).name(), buf, &len, 0);
#endif
    return os;
  }
};

class ELOG_EXPORT eapi_timestamp{
public:
  friend std::ostream& operator<<(std::ostream& os, const eapi_timestamp& rhs) {
    rhs.genstring(os);
    return os;
  }
private:
  void genstring(std::ostream& os) const;
};

template<typename T>
class eapi_printvalue {
public:
  eapi_printvalue() = delete;
  explicit eapi_printvalue(const char* param_name, T data)
    : param_name_(param_name), data_(data) {
  }

  friend std::ostream& operator<<(std::ostream& os,
                                   const eapi_printvalue& rhs) {
    rhs.genstring(os);
    return os;
  }

private:
  void genstring(std::ostream& os) const {
    if (param_name_ == nullptr) return;
    os << param_name_ << ": type=" << eapi_typename<T>()
      << "; value=" << data_ << ";\n";
  }

private:
  const char* param_name_ = nullptr;
  T data_;
};

template<typename T>
class eapi_printarray {
public:
 eapi_printarray() = delete;
 explicit eapi_printarray(T* data, int size = -1) : data_(data), size_(size) {}
 explicit eapi_printarray(const char* param_name, T* data, int size = -1)
     : param_name_(param_name), data_(data), size_(size) {}

 friend std::ostream& operator<<(std::ostream& os, const eapi_printarray& rhs) {
   rhs.genstring(os);
   return os;
  }

private:
  void genstring(std::ostream& os) const {
    if (data_ == nullptr) return;
    int i = 0;
    if(param_name_ != nullptr)
      os << param_name_ << ": type=" << eapi_typename<T>() << "; value=";
    os << "[";
    if (size_ > 0)
      for (i = 0; i < size_; i++) {
        if (i > 0) os << ",";
        os << data_[i];
      }
    else
      while(data_[i] != 0) { // dim shall be non zero
        if (i > 0) os << ",";
        os << data_[i];
        i++;
      }
    os << "];\n";
  }

private:
  const char* param_name_ = nullptr;
  T* data_ = nullptr;
  int size_ = -1;
};

template<typename T>
class eapi_printenum {
public:
  eapi_printenum() = delete;
  explicit eapi_printenum(const char* param_name, T data, const char* type_name)
    : param_name_(param_name), data_(data), type_name_(type_name) {
  }

  friend std::ostream& operator<<(std::ostream& os, const eapi_printenum& rhs) {
    rhs.genstring(os);
    return os;
  }

private:
  void genstring(std::ostream& os) const {
    if (param_name_ == nullptr || type_name_ == nullptr) return;
    os << param_name_ << ": type=" << eapi_typename<T>()
       << "; value=" << type_name_ << " (" << data_ << ");\n";
  }

private:
  const char* param_name_ = nullptr;
  const char* type_name_ = nullptr;
  T data_;
};

#define EAPI_HEAD(module_name, version, func_name)                            \
  module_name << " (" << version << ") function "                             \
  << func_name << "() called:\n"

#define EAPI_TAIL(device_id, handle, stream_id)                                  \
  elog::eapi_timestamp() << "\n"                                              \
  << "Process=" << elog::eapi_get_processid()                                 \
  << ";Thread=" << std::this_thread::get_id()                                 \
  << "; DEVICE=" << (device_id != nullptr ? device_id : "NULL")                        \
  << "; Handle=" << std::showbase << std::uppercase << std::hex               \
  << (handle != nullptr ? handle : "NULL")                                    \
  << "; StreamId=" << (stream_id != nullptr ? stream_id : "NULL")             \
  << ".\n"




#define EAPI_DATA_PAIR(v)                                                     \
  struct data_param_##v {                                                     \
    const char* name;                                                         \
    std::vector<T##v> value;                                                  \
    const char* enum_name = nullptr;                                          \
  } param##v;


#define EAPI_OUTPUT_HEAD \
  os << rhs.module_name << " (" << rhs.verison << ") function "               \
      << rhs.func << "() called:\n";

#define EAPI_OUTPUT(v)                                                        \
  os << rhs.param##v.name << ": type="                                        \
    << eapi_typename<T##v>()                                                  \
    << "; value=";                                                            \
  if (rhs.param##v.value.size() > 1) os << "[";                               \
  else if (rhs.param##v.enum_name != nullptr)                                 \
    os << rhs.param##v.enum_name << " (";                                     \
  for (auto& iter : rhs.param##v.value)                                       \
  {                                                                           \
    if(iter == 0) break; /* bug when non-numerical*/                          \
    os << iter << ",";                                                        \
  }                                                                           \
  os << '\b';                                                                 \
  if (rhs.param##v.value.size() > 1) os << "]";                               \
  else if (rhs.param##v.enum_name != nullptr)                                 \
    os << ")";                                                                \
  os << ";\n";


#define EAPI_OUTPUT_TAIL                                                      \
  auto now = std::chrono::system_clock::now();                                \
  auto du =                                                                   \
    std::chrono::duration_cast<std::chrono::seconds>(now - eapi_time_begin());\
  auto days = du / (24 * 60 * 60);                                            \
  auto hours = (du % (24 * 60 * 60)) / (60 * 60);                             \
  auto minutes = (du % (60 * 60)) / 60;                                       \
  auto seconds = du % 60;                                                     \
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);           \
  auto microseconds =                                                         \
    std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000; \
  os << "Time: "                                                              \
  << std::put_time(std::localtime(&now_time), "%Y-%m-%dT%H:%M:%S.")           \
  << microseconds.count()                                                     \
  << " ("                                                                     \
  << days.count() << "d+"                                                     \
  << hours.count() << "h+"                                                    \
  << minutes.count() << "m+"                                                  \
  << seconds.count() << "s since start)\n"                                    \
  << "Process=" << eapi_get_processid()                                       \
  << ";Thread=" << std::this_thread::get_id()                                 \
  << "; DEVICE=" << (rhs.device_id != nullptr ? rhs.device_id : "NULL");               \
  if(rhs.handle != nullptr)                                                   \
    os << "; Handle=0x" << std::hex << rhs.handle << std::dec;                \
  else                                                                        \
    os << "; Handle=NULL";                                                    \
  if(rhs.stream != nullptr)                                                   \
    os << "; StreamId=" << std::hex << rhs.stream << std::dec;                \
  else                                                                        \
    os << "; StreamId=NULL";                                                  \
  os << ".\n";


template<typename T1, typename ...>
struct eapi_param  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;

  EAPI_DATA_PAIR(1);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2>
struct eapi_param<T1, T2>  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3>
struct eapi_param<T1, T2, T3>  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};


template<typename T1, typename T2, typename T3, typename T4>
struct eapi_param<T1, T2, T3, T4>  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5>
struct eapi_param<T1, T2, T3, T4, T5>  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6>
struct eapi_param<T1, T2, T3, T4, T5, T6>  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7>  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8>  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9>  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10>  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11>  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12>  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12, typename T13>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13>  {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  EAPI_DATA_PAIR(13);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT(13);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12, typename T13, typename T14>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14> {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  EAPI_DATA_PAIR(13);
  EAPI_DATA_PAIR(14);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT(13);
    EAPI_OUTPUT(14);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};


template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12, typename T13, typename T14, typename T15>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
        T15> {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  EAPI_DATA_PAIR(13);
  EAPI_DATA_PAIR(14);
  EAPI_DATA_PAIR(15);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT(13);
    EAPI_OUTPUT(14);
    EAPI_OUTPUT(15);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12, typename T13, typename T14, typename T15,
  typename T16>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
        T15, T16> {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  EAPI_DATA_PAIR(13);
  EAPI_DATA_PAIR(14);
  EAPI_DATA_PAIR(15);
  EAPI_DATA_PAIR(16);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT(13);
    EAPI_OUTPUT(14);
    EAPI_OUTPUT(15);
    EAPI_OUTPUT(16);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12, typename T13, typename T14, typename T15,
  typename T16, typename T17>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
        T15, T16, T17> {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  EAPI_DATA_PAIR(13);
  EAPI_DATA_PAIR(14);
  EAPI_DATA_PAIR(15);
  EAPI_DATA_PAIR(16);
  EAPI_DATA_PAIR(17);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT(13);
    EAPI_OUTPUT(14);
    EAPI_OUTPUT(15);
    EAPI_OUTPUT(16);
    EAPI_OUTPUT(17);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};


template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12, typename T13, typename T14, typename T15,
  typename T16, typename T17, typename T18>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
        T15, T16, T17, T18> {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  EAPI_DATA_PAIR(13);
  EAPI_DATA_PAIR(14);
  EAPI_DATA_PAIR(15);
  EAPI_DATA_PAIR(16);
  EAPI_DATA_PAIR(17);
  EAPI_DATA_PAIR(18);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT(13);
    EAPI_OUTPUT(14);
    EAPI_OUTPUT(15);
    EAPI_OUTPUT(16);
    EAPI_OUTPUT(17);
    EAPI_OUTPUT(18);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12, typename T13, typename T14, typename T15,
  typename T16, typename T17, typename T18, typename T19>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
        T15, T16, T17, T18, T19> {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  EAPI_DATA_PAIR(13);
  EAPI_DATA_PAIR(14);
  EAPI_DATA_PAIR(15);
  EAPI_DATA_PAIR(16);
  EAPI_DATA_PAIR(17);
  EAPI_DATA_PAIR(18);
  EAPI_DATA_PAIR(19);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT(13);
    EAPI_OUTPUT(14);
    EAPI_OUTPUT(15);
    EAPI_OUTPUT(16);
    EAPI_OUTPUT(17);
    EAPI_OUTPUT(18);
    EAPI_OUTPUT(19);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};


template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12, typename T13, typename T14, typename T15,
  typename T16, typename T17, typename T18, typename T19, typename T20>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
        T15, T16, T17, T18, T19, T20> {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  EAPI_DATA_PAIR(13);
  EAPI_DATA_PAIR(14);
  EAPI_DATA_PAIR(15);
  EAPI_DATA_PAIR(16);
  EAPI_DATA_PAIR(17);
  EAPI_DATA_PAIR(18);
  EAPI_DATA_PAIR(19);
  EAPI_DATA_PAIR(20);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT(13);
    EAPI_OUTPUT(14);
    EAPI_OUTPUT(15);
    EAPI_OUTPUT(16);
    EAPI_OUTPUT(17);
    EAPI_OUTPUT(18);
    EAPI_OUTPUT(19);
    EAPI_OUTPUT(20);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};


template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12, typename T13, typename T14, typename T15,
  typename T16, typename T17, typename T18, typename T19, typename T20,
  typename T21>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
        T15, T16, T17, T18, T19, T20, T21> {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  EAPI_DATA_PAIR(13);
  EAPI_DATA_PAIR(14);
  EAPI_DATA_PAIR(15);
  EAPI_DATA_PAIR(16);
  EAPI_DATA_PAIR(17);
  EAPI_DATA_PAIR(18);
  EAPI_DATA_PAIR(19);
  EAPI_DATA_PAIR(20);
  EAPI_DATA_PAIR(21);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT(13);
    EAPI_OUTPUT(14);
    EAPI_OUTPUT(15);
    EAPI_OUTPUT(16);
    EAPI_OUTPUT(17);
    EAPI_OUTPUT(18);
    EAPI_OUTPUT(19);
    EAPI_OUTPUT(20);
    EAPI_OUTPUT(21);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};


template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12, typename T13, typename T14, typename T15,
  typename T16, typename T17, typename T18, typename T19, typename T20,
  typename T21, typename T22>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
        T15, T16, T17, T18, T19, T20, T21, T22> {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  EAPI_DATA_PAIR(13);
  EAPI_DATA_PAIR(14);
  EAPI_DATA_PAIR(15);
  EAPI_DATA_PAIR(16);
  EAPI_DATA_PAIR(17);
  EAPI_DATA_PAIR(18);
  EAPI_DATA_PAIR(19);
  EAPI_DATA_PAIR(20);
  EAPI_DATA_PAIR(21);
  EAPI_DATA_PAIR(22);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT(13);
    EAPI_OUTPUT(14);
    EAPI_OUTPUT(15);
    EAPI_OUTPUT(16);
    EAPI_OUTPUT(17);
    EAPI_OUTPUT(18);
    EAPI_OUTPUT(19);
    EAPI_OUTPUT(20);
    EAPI_OUTPUT(21);
    EAPI_OUTPUT(22);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};


template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12, typename T13, typename T14, typename T15,
  typename T16, typename T17, typename T18, typename T19, typename T20,
  typename T21, typename T22, typename T23>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
        T15, T16, T17, T18, T19, T20, T21, T22, T23> {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  EAPI_DATA_PAIR(13);
  EAPI_DATA_PAIR(14);
  EAPI_DATA_PAIR(15);
  EAPI_DATA_PAIR(16);
  EAPI_DATA_PAIR(17);
  EAPI_DATA_PAIR(18);
  EAPI_DATA_PAIR(19);
  EAPI_DATA_PAIR(20);
  EAPI_DATA_PAIR(21);
  EAPI_DATA_PAIR(22);
  EAPI_DATA_PAIR(23);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT(13);
    EAPI_OUTPUT(14);
    EAPI_OUTPUT(15);
    EAPI_OUTPUT(16);
    EAPI_OUTPUT(17);
    EAPI_OUTPUT(18);
    EAPI_OUTPUT(19);
    EAPI_OUTPUT(20);
    EAPI_OUTPUT(21);
    EAPI_OUTPUT(22);
    EAPI_OUTPUT(23);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};


template<typename T1, typename T2, typename T3, typename T4, typename T5,
  typename T6, typename T7, typename T8, typename T9, typename T10,
  typename T11, typename T12, typename T13, typename T14, typename T15,
  typename T16, typename T17, typename T18, typename T19, typename T20,
  typename T21, typename T22, typename T23, typename T24>
struct eapi_param<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14,
        T15, T16, T17, T18, T19, T20, T21, T22, T23, T24> {
  const char* module_name;
  const char* verison;
  const char* func;
  const char* device_id;
  const void* handle;
  const void* stream;
  EAPI_DATA_PAIR(1);
  EAPI_DATA_PAIR(2);
  EAPI_DATA_PAIR(3);
  EAPI_DATA_PAIR(4);
  EAPI_DATA_PAIR(5);
  EAPI_DATA_PAIR(6);
  EAPI_DATA_PAIR(7);
  EAPI_DATA_PAIR(8);
  EAPI_DATA_PAIR(9);
  EAPI_DATA_PAIR(10);
  EAPI_DATA_PAIR(11);
  EAPI_DATA_PAIR(12);
  EAPI_DATA_PAIR(13);
  EAPI_DATA_PAIR(14);
  EAPI_DATA_PAIR(15);
  EAPI_DATA_PAIR(16);
  EAPI_DATA_PAIR(17);
  EAPI_DATA_PAIR(18);
  EAPI_DATA_PAIR(19);
  EAPI_DATA_PAIR(20);
  EAPI_DATA_PAIR(21);
  EAPI_DATA_PAIR(22);
  EAPI_DATA_PAIR(23);
  EAPI_DATA_PAIR(24);
  friend std::ostream& operator << (std::ostream& os, const eapi_param& rhs) {
    EAPI_OUTPUT_HEAD;
    EAPI_OUTPUT(1);
    EAPI_OUTPUT(2);
    EAPI_OUTPUT(3);
    EAPI_OUTPUT(4);
    EAPI_OUTPUT(5);
    EAPI_OUTPUT(6);
    EAPI_OUTPUT(7);
    EAPI_OUTPUT(8);
    EAPI_OUTPUT(9);
    EAPI_OUTPUT(10);
    EAPI_OUTPUT(11);
    EAPI_OUTPUT(12);
    EAPI_OUTPUT(13);
    EAPI_OUTPUT(14);
    EAPI_OUTPUT(15);
    EAPI_OUTPUT(16);
    EAPI_OUTPUT(17);
    EAPI_OUTPUT(18);
    EAPI_OUTPUT(19);
    EAPI_OUTPUT(20);
    EAPI_OUTPUT(21);
    EAPI_OUTPUT(22);
    EAPI_OUTPUT(23);
    EAPI_OUTPUT(24);
    EAPI_OUTPUT_TAIL;
    return os;
  }
};

}
