#pragma once


#include "elog/elog.h"
#include "spdlog/spdlog.h"

namespace elog {

ELOG_EXPORT std::shared_ptr<spdlog::logger> get_spdlog_instance();
ELOG_EXPORT spdlog::level::level_enum get_spdlog_level(level lvl);

template<typename... Args>
void efmt(int handle, level lvl, const char* format, const char* filename, int line, Args &&... args) {
  if (lvl > get_level(handle))
    return;

#if defined(_MSC_VER)
  const char* base = strrchr(filename, '\\');
#else
  const char* base = strrchr(filename, '/');
#endif
  const char* basename = base ? (base + 1) : filename;

  get_spdlog_instance()->log(get_spdlog_level(lvl), format, basename, line, std::forward<Args>(args)...);
}


template<typename... Args>
void efmt(const char* modulename, level lvl, const char* format, const char* filename, int line, Args &&... args) {
  int handle = get_module_handle(modulename);
  if (handle < 0)
    return;
  efmt(handle, lvl, format, filename, line, std::forward<Args>(args)...);
}

} // namespace elog


#define EFMT(handle, lvl, fmt, ...) \
  if(elog::get_level(handle) >= lvl) { \
    elog::efmt(handle, lvl, "{}:{}: "##fmt, __FILE__, __LINE__, ##__VA_ARGS__); \
  }



