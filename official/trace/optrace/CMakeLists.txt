cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(CMAKE_COMPILER_IS_GNUCXX)
  add_compile_options(-Wall -Wno-switch -Wextra -Wpedantic)
endif()

project(optrace VERSION 1.0.0)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

set(OPTRACE_SOURCE_FILES
  src/optrace.cc
)

add_library(optrace_static STATIC ${OPTRACE_SOURCE_FILES})
target_compile_options(optrace_static PRIVATE -g -O0)
target_include_directories(optrace_static PUBLIC src)

add_library(optrace SHARED ${OPTRACE_SOURCE_FILES})
target_compile_options(optrace PRIVATE -g -O0)
target_include_directories(optrace PUBLIC src)
set_target_properties(optrace PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
)


add_subdirectory(3rd/pybind11)
set(OPTRACE_PYBIND_SOURCE_FILES
  src/optrace.cc
  pybind11/optrace_pybind.cc
)
pybind11_add_module(optrace_py ${OPTRACE_PYBIND_SOURCE_FILES})
target_compile_options(optrace_py PRIVATE -g -O0)
target_include_directories(optrace_py PUBLIC src)

add_library(optrace::optrace INTERFACE IMPORTED GLOBAL)
target_include_directories(optrace::optrace INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/>)
target_link_libraries(optrace::optrace INTERFACE optrace_static)

add_executable(optrace_sample sample/sample.cc)
target_compile_options(optrace_sample PRIVATE -g -O0)
target_link_libraries(optrace_sample PRIVATE optrace::optrace)
