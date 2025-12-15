#!/bin/bash

set -euo pipefail

DIR=$(dirname $(readlink -f $0))
echo ${DIR}
cd ${DIR}/

rm -fr cpp

header_name=cudnn_ops
sed 's/\bCUDNN_DEPRECATED\b//g' ../header/include/x86_64-linux-gnu/${header_name}.h > ${header_name}.h
python3 gen.py -t cudnn -f ${header_name}.h -o cpp
rm ${header_name}.h

header_name=cudnn_cnn
sed 's/\bCUDNN_DEPRECATED\b//g' ../header/include/x86_64-linux-gnu/${header_name}.h > ${header_name}.h
python3 gen.py -t cudnn -f ${header_name}.h -o cpp
rm ${header_name}.h

header_name=cudnn_graph
sed 's/\bCUDNN_DEPRECATED\b//g' ../header/include/x86_64-linux-gnu/${header_name}.h > ${header_name}.1
sed 's/\CUDNN_DEPRECATED_ENUM\b//g' ${header_name}.1 > ${header_name}.h
python3 gen.py -t cudnn -f ${header_name}.h -o cpp
rm ${header_name}.1
rm ${header_name}.h


header_name=cudnn_adv
python3 gen.py -t cudnn -f ../header/include/x86_64-linux-gnu/${header_name}.h -o cpp


#clange_format.sh
