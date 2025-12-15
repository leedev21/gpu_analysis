// Copyright 2022. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 14:54:28 on Sun, May 29, 2022
//
// Description: trace and profile

#ifndef __CUDA_HOOK_TRACE_PROFILE_H__
#define __CUDA_HOOK_TRACE_PROFILE_H__

#include <chrono>
#include <string>

#include "macro_common.h"
#include "log.h"

__inline__ int getIdx() {
  static int idx = 0;
  return idx++;
}

class TraceProfile {
 public:
  TraceProfile(int enable, const std::string &name)
      : enable_(enable),
        m_name(name),
        m_start(std::chrono::steady_clock::now()) {
    if (enable) {
      HLOGEX("%s enter\n", m_name.c_str());
    }
  }

  ~TraceProfile() {
    m_end = std::chrono::steady_clock::now();
    m_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start);
    if (enable_) {
      HLOGEX("%s exit, cost %f ms\n", m_name.c_str(), m_duration.count());
    }
  }

 private:
  const std::string m_name;
  std::chrono::steady_clock::time_point m_start;
  std::chrono::steady_clock::time_point m_end;
  std::chrono::duration<double, std::milli> m_duration;
  int enable_ = 0;

  HOOK_DISALLOW_COPY_AND_ASSIGN(TraceProfile);
};

#define HOOK_TRACE_PROFILE_ALWAYS(name) \
  TraceProfile _tp_##name_(             \
      name);  // std::cout << " func idx " << getIdx() << std::endl;

#define HOOK_TRACE_PROFILE(en, name) TraceProfile _tp_##name_(en, name);

#endif  // __CUDA_HOOK_TRACE_PROFILE_H__
