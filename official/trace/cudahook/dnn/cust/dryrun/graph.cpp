#include "cudnn_graph.h"
#include "common/hook.h"
#include "common/api_log.h"
#include "common/black_list.h"
#include "common/utils.hpp"
#include "dnn/cust/declears.h"

namespace hook {
struct MyHandle_t {
  cudaStream_t stream_ = nullptr;
};
cudnnStatus_t cudnnBackendCreateDescriptor(
    cudnnBackendDescriptorType_t descriptorType,
    cudnnBackendDescriptor_t *descriptor) {
  using func_ptr = cudnnStatus_t (*)(cudnnBackendDescriptorType_t,
                                     cudnnBackendDescriptor_t *);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDNN_SYMBOL("cudnnBackendCreateDescriptor"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(descriptorType, descriptor);
  return _ret;
}

cudnnStatus_t cudnnBackendDestroyDescriptor(
    cudnnBackendDescriptor_t descriptor) {
  using func_ptr = cudnnStatus_t (*)(cudnnBackendDescriptor_t);
  static auto func_entry = reinterpret_cast<func_ptr>(
      HOOK_CUDNN_SYMBOL("cudnnBackendDestroyDescriptor"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(descriptor);
  return _ret;
}

cudnnStatus_t cudnnBackendGetAttribute(
    cudnnBackendDescriptor_t const descriptor,
    cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, int64_t requestedElementCount,
    int64_t *elementCount, void *arrayOfElements) {
  using func_ptr = cudnnStatus_t (*)(
      cudnnBackendDescriptor_t const, cudnnBackendAttributeName_t,
      cudnnBackendAttributeType_t, int64_t, int64_t *, void *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnBackendGetAttribute"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(descriptor, attributeName, attributeType,
                         requestedElementCount, elementCount, arrayOfElements);
  return _ret;
}

cudnnStatus_t cudnnBackendSetAttribute(
    cudnnBackendDescriptor_t descriptor,
    cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType, int64_t elementCount,
    const void *arrayOfElements) {
  using func_ptr =
      cudnnStatus_t (*)(cudnnBackendDescriptor_t, cudnnBackendAttributeName_t,
                        cudnnBackendAttributeType_t, int64_t, const void *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnBackendSetAttribute"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(descriptor, attributeName, attributeType, elementCount,
                         arrayOfElements);
  return _ret;
}

cudnnStatus_t cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor) {
  using func_ptr = cudnnStatus_t (*)(cudnnBackendDescriptor_t);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnBackendFinalize"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(descriptor);
  return _ret;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t *streamId) {
  using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudaStream_t *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnGetStream"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(handle, streamId);
  return _ret;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
  using func_ptr = cudnnStatus_t (*)(cudnnHandle_t);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnDestroy"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(handle);
  return _ret;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId) {
  using func_ptr = cudnnStatus_t (*)(cudnnHandle_t, cudaStream_t);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnSetStream"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(handle, streamId);
  return _ret;
}

cudnnStatus_t cudnnCreate(cudnnHandle_t *handle) {
  using func_ptr = cudnnStatus_t (*)(cudnnHandle_t *);
  static auto func_entry =
      reinterpret_cast<func_ptr>(HOOK_CUDNN_SYMBOL("cudnnCreate"));
  HOOK_CHECK(func_entry);
  auto _ret = func_entry(handle);
  return _ret;
}

cudnnStatus_t cudnnBackendExecute(cudnnHandle_t, cudnnBackendDescriptor_t,
                                  cudnnBackendDescriptor_t) {
  return CUDNN_STATUS_SUCCESS;
}

size_t cudnnGetVersion(void) { return 90000; }

const char *cudnnGetErrorString(cudnnStatus_t) {
  static const char *str = "fake_dnn_error_str";
  return str;
}

DRYRUN_REGISTE_FUNC(cudnnBackendExecute)
DRYRUN_REGISTE_FUNC(cudnnGetErrorString)
DRYRUN_REGISTE_FUNC(cudnnGetVersion)

DRYRUN_REGISTE_FUNC(cudnnCreate)

DRYRUN_REGISTE_FUNC(cudnnBackendFinalize)
DRYRUN_REGISTE_FUNC(cudnnBackendSetAttribute)
DRYRUN_REGISTE_FUNC(cudnnGetStream)
DRYRUN_REGISTE_FUNC(cudnnDestroy)
DRYRUN_REGISTE_FUNC(cudnnSetStream)

DRYRUN_REGISTE_FUNC(cudnnBackendCreateDescriptor)
DRYRUN_REGISTE_FUNC(cudnnBackendDestroyDescriptor)
DRYRUN_REGISTE_FUNC(cudnnBackendGetAttribute)
}  // namespace hook
