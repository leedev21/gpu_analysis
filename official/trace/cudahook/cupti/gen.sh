#!/bin/bash

set -euo pipefail

DIR=$(dirname $(readlink -f $0))
echo ${DIR}
cd ${DIR}/

rm -fr cpp

header_name=cupti_events
python3 gen.py -t cupti -f ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h -o cpp


header_name=cupti_metrics
python3 gen.py -t cupti -f ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h -o cpp


header_name=cupti_activity
python3 gen.py -t cupti -f ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h -o cpp


header_name=cupti_callbacks
python3 gen.py -t cupti -f ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h -o cpp


header_name=cupti_result
python3 gen.py -t cupti -f ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h -o cpp


header_name=cupti_version
python3 gen.py -t cupti -f ../header/local/cuda-12.4/targets/x86_64-linux/include/${header_name}.h -o cpp
