// auto generate 22 apis

#include "cudnn_graph.h"
#include "common/hook.h"
#include "common/api_log.h"
#include "dnn/api_log.h"
#include "common/black_list.h"
#include "common/utils.hpp"

namespace hook {
class cudnnGetLastErrorString_PtrProvider {
 private:
  void* raw_ = nullptr;
  void* hook_;

 public:
  cudnnGetLastErrorString_PtrProvider(void* h) {
    cudaHookInit();
    raw_ = HOOK_CUDNN_SYMBOL("cudnnGetLastErrorString");
    hook_ = h;
  }
  ~cudnnGetLastErrorString_PtrProvider() = default;
  void* Get() { return hook_ ? hook_ : raw_; }
  void* GetRaw() { return raw_; }
};
cudnnGetLastErrorString_PtrProvider* Get_cudnnGetLastErrorString_Obj(
    void* hook = nullptr) {
  static cudnnGetLastErrorString_PtrProvider instace(hook);
  return &instace;
}
void* Get_cudnnGetLastErrorString_NativePtr() {
  return Get_cudnnGetLastErrorString_Obj()->GetRaw();
}
void* Get_cudnnGetLastErrorString_Ptr(void* hook = nullptr) {
  return Get_cudnnGetLastErrorString_Obj(hook)->Get();
}
}  // namespace hook

HOOK_C_API HOOK_DECL_EXPORT void cudnnGetLastErrorString(char* message,
                                                         size_t max_size) {
  static bool log_enable = !isApiInBlackList("cudnnGetLastErrorString");
  using func_ptr = void (*)(char*, size_t);
  static auto func_entry =
      reinterpret_cast<func_ptr>(hook::Get_cudnnGetLastErrorString_Ptr());
  HOOK_CHECK(func_entry);
  if (log_enable) {
    ParamDumper inst(__func__);
    inst.Dump(message, max_size);
  }
  return func_entry(message, max_size);
}