# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 16:19:40 on Sun, May 29, 2022
#
# Description: cmake for cuda hook


#set (CMAKE_C_VISIBILITY_PRESET hidden)
#set (CMAKE_CXX_VISIBILITY_PRESET hidden)
#set (CMAKE_POSITION_INDEPENDENT_CODE ON)

#set (CMAKE_C_FLAGS "-std=c11")
#set (CMAKE_C_FLAGS_DEBUG "$ENV{CFLAGS} -O0 -g2 -ggdb")
#set (CMAKE_C_FLAGS_RELEASE "$ENV{CFLAGS} -O3")

#set (CMAKE_CXX_FLAGS "-std=c++11")
#set (CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g2 -ggdb")
#set (CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")

#set (CMAKE_SHARED_LINKER_FLAGS "-s -Wl,--exclude-libs,ALL")
#set (CMAKE_EXE_LINKER_FLAGS "-Wl,--as-needed")

set (SO_NAME cudart)

add_compile_options (
    #-Wall
    #-Werror
    -Wextra
    -Wswitch-default
    -Wfloat-equal
    -Wshadow
    -Wcast-qual
)

include_directories (  
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/header/local/cuda-12.4/targets/x86_64-linux/include/
)

#file (GLOB          HOOK_SRCS     cpp/*.cpp)
file (GLOB_RECURSE  HOOK_SRCS     *.cpp)
file (GLOB_RECURSE  COMMON_SRCS   ${PROJECT_SOURCE_DIR}/common/*.cpp)


add_library(${SO_NAME} SHARED ${HOOK_SRCS} ${COMMON_SRCS})

target_link_libraries(${SO_NAME} -ldl libilog.a)

install(TARGETS ${SO_NAME} LIBRARY DESTINATION ./)
