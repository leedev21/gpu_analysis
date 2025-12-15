# Copyright 2022. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 23:56:07 on Sat, May 28, 2022
#
# Description: code generate script

#!/bin/bash

set -euo pipefail

DIR=$(dirname $(readlink -f $0))
echo ${DIR}
cd ${DIR}/

rm -fr cpp

#header_name=cublas_api
#python3 code_generate.py -t cublasLt -f /usr/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h -o output

header_name=cublasLt
python3 ./gen.py -t cublasLt -f /usr/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h -o cpp

#clange_format.sh
