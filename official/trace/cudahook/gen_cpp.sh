DIR=$(dirname $(readlink -f $0))
echo ${DIR}
cd ${DIR}

echo "begin parse head and generate cpp files:"

cupti/gen.sh
cufft/gen.sh
cuda/gen.sh
cudart/gen.sh
cublas/gen.sh
dnn/gen.sh
cusparse/gen.sh
curand/gen.sh
nccl/gen.sh
nvml/gen.sh

echo "finish parse head and generate cpp files"
