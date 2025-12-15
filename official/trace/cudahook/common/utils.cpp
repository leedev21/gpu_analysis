#include <thread>
#include <mutex>
#include <unistd.h>

#include "common/log.h"
#include "common/env_default.hpp"

thread_local int device_id = 0;
std::mutex device_id_mtx;

int getDeviceId() {
  std::lock_guard<std::mutex> lock(device_id_mtx);
  return device_id;
}
void setDeviceId(int id) {
  std::lock_guard<std::mutex> lock(device_id_mtx);
  device_id = id;
}

class Hold {
 public:
  Hold() {
    if (1 == VALUE(CH_DBG_WAIT10S)) {
      HLOGEX("wait 10s begin\n");
      sleep(10);
    }
  }
  ~Hold() = default;
};

void cudaHookInit() { static Hold instance; }

int getFakeDeviceCount() { return VALUE(CH_VIRT_DEVICE_COUNT); }
