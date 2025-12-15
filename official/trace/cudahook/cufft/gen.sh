#!/bin/bash

set -euo pipefail

DIR=$(dirname $(readlink -f $0))
echo ${DIR}
cd ${DIR}/

rm -fr cpp

header_name=cufft
python3 gen.py -t cufft -f ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h -o cpp


header_name=cufftXt
python3 gen.py -t cufft -f ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h -o cpp

