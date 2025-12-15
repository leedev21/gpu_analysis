set (SO_NAME hook_${DIR})
add_custom_target(
    gen_${SO_NAME}_cpp
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR} && ./gen.sh
    USES_TERMINAL
)
add_compile_options (
    #-Wall
    #-Werror
    -Wextra
    -Wswitch-default
    -Wfloat-equal
    -Wshadow
    -Wcast-qual
)

set (SYS_CUDART_PATH "/usr/local/cuda")

include_directories (
  ${SYS_CUDART_PATH}/include
  ${PROJECT_SOURCE_DIR}
)

#file (GLOB          HOOK_SRCS     cpp/*.cpp)
file (GLOB_RECURSE          HOOK_SRCS     *.cpp)
file (GLOB_RECURSE  COMMON_SRCS   ${PROJECT_SOURCE_DIR}/common/*.cpp)

add_library(${SO_NAME} SHARED ${HOOK_SRCS} ${COMMON_SRCS})

target_link_libraries(${SO_NAME} -ldl libilog.a)
add_dependencies(${SO_NAME} gen_${SO_NAME}_cpp)

install(TARGETS ${SO_NAME} LIBRARY DESTINATION ./)
