#define NVML_NO_UNVERSIONED_FUNC_DEFS

// auto generate 345 apis

#include "nvml.h"
#include "common/hook.h"
#include "common/macro_common.h"
#include "common/api_log.h"
#include "common/utils.hpp"
#include "common/black_list.h"
#include "nvml/error_check.h"
#include "nvml/cust/declares.h"

namespace hook {
namespace dryrun {

nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *deviceCount) {
  if (deviceCount) {
    *deviceCount = getFakeDeviceCount();
  }
  return NVML_SUCCESS;
}

nvmlReturn_t nvmlInit() { return NVML_SUCCESS; }
nvmlReturn_t nvmlInit_v2() { return NVML_SUCCESS; }

DRYRUN_REGISTE_FUNC(nvmlDeviceGetCount_v2)
DRYRUN_REGISTE_FUNC(nvmlInit)
DRYRUN_REGISTE_FUNC(nvmlInit_v2)

}  // namespace dryrun
}  // namespace hook