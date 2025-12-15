#!/bin/bash

set -euo pipefail

DIR=$(dirname $(readlink -f $0))
echo ${DIR}
cd ${DIR}/

rm -fr cpp
header_name=cublas_api
python3 gen.py -t cublas -f ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h -o cpp
sed -i '1i #define CUBLASAPI' cpp/${header_name}.h_hook.cpp


## blasLt
header_name=cublasLt
python3 gen_cublasLt.py -t cublasLt -f ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h -o cpp

