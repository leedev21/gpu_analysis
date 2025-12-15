#pragma once

#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <string>

#include "elog/elog.h"
#include "common/utils.hpp"

// #define CH_LOG FCOUT(elog::info)
//#define CH_LOG FCOUT(elog::info) << getDeviceId() << ":-1 "

inline char *curr_time1() {
  time_t raw_time = time(nullptr);
  struct tm *time_info = localtime(&raw_time);
  static char now_time[64];
  now_time[strftime(now_time, sizeof(now_time), "%Y-%m-%d %H:%M:%S",
                    time_info)] = '\0';
  return now_time;
}

inline int get_pid1() {
  static int pid = getpid();
  return pid;
}

inline long int get_tid1() {
  thread_local long int tid = syscall(SYS_gettid);
  return tid;
}

#define HLOGEX(format, ...)                                              \
  do {                                                                   \
    fprintf(stdout, "%s cuda_api: A%d: %d:-1 " format "\n", curr_time1(), \
            get_pid1(), getDeviceId(), ##__VA_ARGS__);                    \
  } while (0)

// #define HLOGEX(format, ...)                                           \
//   do {                                                                \
//     fprintf(stdout, "%s %d:%ld " format "\n", curr_time1(), get_pid1(), \
//             get_tid1(), ##__VA_ARGS__);                                \
//   } while (0)
