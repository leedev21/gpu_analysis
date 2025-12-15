set (SO_NAME cublas)

add_compile_options (
    #-Wall
    #-Werror
    -Wextra
    -Wswitch-default
    -Wfloat-equal
    -Wshadow
    -Wcast-qual
)

# set (SYS_CUDART_PATH "/usr/local/cuda")

include_directories (
  # ${SYS_CUDART_PATH}/include
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/header/local/cuda-12.4/targets/x86_64-linux/include/
)

#file (GLOB          HOOK_SRCS     cpp/*.cpp)
file (GLOB_RECURSE          HOOK_SRCS     *.cpp)
file (GLOB_RECURSE  COMMON_SRCS   ${PROJECT_SOURCE_DIR}/common/*.cpp)

add_library(${SO_NAME} SHARED ${HOOK_SRCS} ${COMMON_SRCS})

target_link_libraries(${SO_NAME} -ldl libilog.a)

install(TARGETS ${SO_NAME} LIBRARY DESTINATION ./)
