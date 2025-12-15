#!/bin/bash

set -euo pipefail

DIR=$(dirname $(readlink -f $0))
echo ${DIR}
cd ${DIR}/

rm -fr cpp

header_name=nvml

grep -v "^#include" ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h >${header_name}.2.h
gcc -DNVML_NO_UNVERSIONED_FUNC_DEFS -E ${header_name}.2.h >${header_name}.1.h
sed '/^\s*#/d' ${header_name}.1.h >${header_name}.h
python3 gen.py -t nvml -f ${header_name}.h -o cpp  
sed -i '1i #define NVML_NO_UNVERSIONED_FUNC_DEFS' cpp/${header_name}.h_hook.cpp
rm ${header_name}.h
rm ${header_name}.1.h
rm ${header_name}.2.h
