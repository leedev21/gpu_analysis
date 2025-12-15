#pragma once
#include <vector>
#include <string>

#define VALUE(name) _GetEnv_##name()
#define DECLARE_flag(type, name) extern type& _GetEnv_##name()
#define DECLARE_bool(name) DECLARE_flag(bool, name)
#define DECLARE_int64(name) DECLARE_flag(int64_t, name)

#define DECLARE_StringArray(name) DECLARE_flag(std::vector<std::string>, name)

DECLARE_int64(CH_WORK_MODE);

DECLARE_StringArray(CH_API_BLACKLIST);

DECLARE_int64(CH_DBG_DUMP_KERNEL_NAME);

DECLARE_int64(CH_DBG_WAIT10S);

DECLARE_int64(CH_VIRT_DEVICE_ID);//
DECLARE_int64(CH_VIRT_DEVICE_COUNT);//


