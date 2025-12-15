#include <iostream>
#include <vector>
#include <string>
#include <regex>
#include <algorithm>

namespace {
bool GetBoolFromEnv(const std::string& envVarName, bool defaultValue) {
  const char* envValue = std::getenv(envVarName.c_str());
  if (envValue != nullptr) {
    std::string valueStr(envValue);
    if (valueStr == "1" || valueStr == "true" || valueStr == "TRUE" ||
        valueStr == "True") {
      return true;
    } else if (valueStr == "0" || valueStr == "false" || valueStr == "FALSE" ||
               valueStr == "False") {
      return false;
    } else {
      std::cout << "get a wrong value from env:" << envVarName << " "
                << valueStr << ", \nthen use the default " << defaultValue
                << std::endl;
    }
  }
  return defaultValue;
}

int64_t GetInt64FromEnv(const std::string& envVarName, int64_t defaultValue) {
  const char* envValue = std::getenv(envVarName.c_str());
  if (envValue != nullptr) {
    try {
      int64_t value = 0;
      if (envVarName == "DEVICE") {
        value = std::stoll(envValue, nullptr, 2);
      } else {
        value = std::stoll(envValue, nullptr);
      }
      // std::cout << "set " << envVarName << " to " << value << std::endl;
      return value;
    } catch (const std::exception& e) {
      std::cout << "get an exception from env:" << envVarName
                << ", \nthen use the default " << defaultValue << std::endl;
    }
  }
  return defaultValue;
}

std::vector<std::string> GetStrArray(const char* _var) {
  std::vector<std::string> result;
  do {
    const char* var = std::getenv(_var);
    if (!var) {
      break;
    }

    // remove blank space
    std::string str(var);
    str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());

    //
    std::regex re("([^:]+)");
    std::smatch match;
    std::string::const_iterator searchStart(str.cbegin());

    // 使用正则表达式提取所有子串
    while (std::regex_search(searchStart, str.cend(), match, re)) {
      result.push_back(match[1]);
      searchStart = match.suffix().first;
    }

    // for (const auto& it : result) {
    //   std::cout<< __func__ << "   " << it << std::endl;
    // }
  } while (0);
  return result;
}

}  // namespace

#define DEFINE_flag(type, name, deflt, desc) \
  type& _GetEnv_##name() {                   \
    static type inst = deflt;                \
    return inst;                             \
  }

#define DEFINE_bool(name, deflt, desc) \
  DEFINE_flag(bool, name, GetBoolFromEnv(#name, deflt), desc)

#define DEFINE_int64(name, deflt, desc) \
  DEFINE_flag(int64_t, name, GetInt64FromEnv(#name, deflt), desc)

#define DEFINE_StringArray(name, desc) \
  DEFINE_flag(std::vector<std::string>, name, GetStrArray(#name), desc)

DEFINE_int64(CH_WORK_MODE, (0), "0 for native mode, 1 for dry run");

DEFINE_StringArray(CH_API_BLACKLIST, "api black list");

DEFINE_int64(CH_DBG_DUMP_KERNEL_NAME, (1), "debug dump kernel name");
DEFINE_int64(CH_DBG_WAIT10S, (0), "debug wait 10s");
DEFINE_int64(CH_VIRT_DEVICE_ID, (0), "VIRT_DEVICE_ID, 0--3060 1--V100 2--A100 3--H100");
DEFINE_int64(CH_VIRT_DEVICE_COUNT, (16), "VIRT_DEVICE_COUNT");


