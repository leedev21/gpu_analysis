#!/bin/bash

set -euo pipefail

DIR=$(dirname $(readlink -f $0))
echo ${DIR}
cd ${DIR}/

rm -fr cpp

header_name=cusparse
sed 's/\CUSPARSE_DEPRECATED_TYPE\b//g' ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h > ${header_name}.1
sed 's/\CUSPARSE_DEPRECATED\b//g' ${header_name}.1 > ${header_name}.h
python3 gen.py -t cusparse -f ${header_name}.h -o cpp
rm ${header_name}.1
rm ${header_name}.h
