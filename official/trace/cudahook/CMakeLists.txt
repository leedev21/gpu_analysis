cmake_minimum_required(VERSION 3.22)

project(CudaHook LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/bin")

add_compile_options(-Wno-attributes)

include_directories (  
  ${PROJECT_SOURCE_DIR}/
  ${PROJECT_SOURCE_DIR}/2nd/
  ${PROJECT_SOURCE_DIR}/header/include/
  ${PROJECT_SOURCE_DIR}/header/local/cuda-12.4/targets/x86_64-linux/include/
)
link_directories(${PROJECT_SOURCE_DIR}/2nd/elog/)


add_subdirectory(cupti)
add_subdirectory(cufft)
add_subdirectory(cuda)
add_subdirectory(cudart)
add_subdirectory(cublas)
add_subdirectory(dnn)
add_subdirectory(cusparse)
add_subdirectory(curand)
add_subdirectory(nccl)
add_subdirectory(nvml)

#add_subdirectory(cublasLt)
