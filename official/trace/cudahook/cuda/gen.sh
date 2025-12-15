#!/bin/bash

set -euo pipefail

DIR=$(dirname $(readlink -f $0))
echo ${DIR}
cd ${DIR}/

rm -fr cpp

header_name=cuda
grep -v "^#include" ../header/local/cuda-12.4/targets/x86_64-linux/include//${header_name}.h >${header_name}.1.h
gcc -E -D__CUDA_API_VERSION_INTERNAL -D__CUDA_API_PER_THREAD_DEFAULT_STREAM -E ${header_name}.1.h > ${header_name}.2.h
sed '/_Alignas(64)/d' ${header_name}.2.h > ${header_name}.3.h
sed '/^\s*#/d' ${header_name}.3.h > ${header_name}.h
python3 gen.py -t cuda -f ${header_name}.h -o cpp
sed -i '1i #define __CUDA_API_VERSION_INTERNAL' cpp/${header_name}.h_hook.cpp
sed -i '1i #define __CUDA_API_PER_THREAD_DEFAULT_STREAM' cpp/${header_name}.h_hook.cpp
rm ${header_name}.1.h
rm ${header_name}.2.h
rm ${header_name}.3.h
rm ${header_name}.h

