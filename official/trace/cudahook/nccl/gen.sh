#!/bin/bash

set -euo pipefail

DIR=$(dirname $(readlink -f $0))
echo ${DIR}
cd ${DIR}/

rm -fr cpp

header_name=nccl
grep -v "^#include" ../header/include/${header_name}.h >${header_name}.2.h
gcc -D__CUDA_BF16_TYPES_EXIST__  -E ${header_name}.2.h >${header_name}.1.h
sed '/^\s*#/d' ${header_name}.1.h >${header_name}.h
python3 gen.py -t nccl -f ${header_name}.h -o cpp  
rm ${header_name}.h
rm ${header_name}.1.h
rm ${header_name}.2.h


#clange_format.sh
