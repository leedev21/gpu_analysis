#!/bin/bash

set -euo pipefail

DIR=$(dirname $(readlink -f $0))
echo ${DIR}
cd ${DIR}/

rm -fr cpp

header_name=cuda_runtime_api
grep -v "^#include" ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h >${header_name}.1.h
gcc -E ${header_name}.1.h >${header_name}.2.h
sed '/^\s*#/d' ${header_name}.2.h >${header_name}.h
python3 gen.py -t cudart -f ${header_name}.h -o cpp
sed -i '1i #define __CUDA_API_VERSION_INTERNAL' cpp/${header_name}.h_hook.cpp
sed -i '1i #define __CUDA_API_PER_THREAD_DEFAULT_STREAM' cpp/${header_name}.h_hook.cpp
sed -i '1i #define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__' cpp/${header_name}.h_hook.cpp
rm ${header_name}.1.h
rm ${header_name}.2.h
rm ${header_name}.h


header_name=cuda_profiler_api
grep -v "^#include" ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h >${header_name}.2.h
gcc -E ${header_name}.2.h >${header_name}.1.h
sed '/^\s*#/d' ${header_name}.1.h >${header_name}.h
python3 gen.py -t cudart -f ${header_name}.h -o cpp
sed -i '1i #define __CUDA_API_VERSION_INTERNAL' cpp/${header_name}.h_hook.cpp
sed -i '1i #define __CUDA_API_PER_THREAD_DEFAULT_STREAM' cpp/${header_name}.h_hook.cpp
sed -i '1i #define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__' cpp/${header_name}.h_hook.cpp
rm ${header_name}.h
rm ${header_name}.1.h
rm ${header_name}.2.h



header_name=host_runtime
grep -v "^#include" ../header/local/cuda-12.4/targets/x86_64-linux/include/crt/${header_name}.h >${header_name}.2.h
gcc -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__  -D__CUDA_INTERNAL_COMPILATION__ -E ${header_name}.2.h >${header_name}.1.h
sed '/^\s*#/d' ${header_name}.1.h >${header_name}.h
python3 gen_host_runtime.py -t cudart -f ${header_name}.h -o cpp
sed 's/(void ( \* ) ( void \* \* ) v, void \* vv, void \* vvv, void ( \* ) ( void \* ) vvvv)/(void ( \* v ) ( void \*\* ), void \* vv, void \* vvv, void ( \* vvvv ) ( void \* ))/g' cpp/${header_name}.h_hook.cpp > cpp/${header_name}.h.cpp.1
sed 's/atexit(void ( \* ) ( void ) v)/atexit(void ( \* v ) ( void ) )/g' cpp/${header_name}.h.cpp.1 > cpp/${header_name}.h.cpp

sed -i '1i #define __CUDA_API_VERSION_INTERNAL' cpp/${header_name}.h.cpp
sed -i '1i #define __CUDA_API_PER_THREAD_DEFAULT_STREAM' cpp/${header_name}.h.cpp
sed -i '1i #define __CUDACC_RTC__' cpp/${header_name}.h.cpp

rm ${header_name}.1.h
rm ${header_name}.2.h
rm ${header_name}.h
rm cpp/${header_name}.h_hook.cpp
rm cpp/${header_name}.h.cpp.1



header_name=device_functions
grep -v "^#include" ../header/local/cuda-12.4/targets/x86_64-linux/include/crt/${header_name}.h >${header_name}.2.h
gcc -D__CUDACC__ -D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__ -E ${header_name}.2.h >${header_name}.1.h
sed '/^\s*#/d' ${header_name}.1.h >${header_name}.h
python3 gen_host_runtime.py -t cudart -f ${header_name}.h -o cpp

sed -i '1i #define __CUDA_API_VERSION_INTERNAL' cpp/${header_name}.h_hook.cpp
sed -i '1i #define __CUDA_API_PER_THREAD_DEFAULT_STREAM' cpp/${header_name}.h_hook.cpp
sed -i '1i #define __CUDACC_RTC__' cpp/${header_name}.h_hook.cpp
rm ${header_name}.h
rm ${header_name}.1.h
rm ${header_name}.2.h






