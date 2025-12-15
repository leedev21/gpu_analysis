#!/bin/bash
set -eu -o pipefail

DIR=$(dirname $(readlink -f $0))
echo ${DIR}
cd ${DIR}
./gen_cpp.sh

cd ${DIR}
rm -rf cmake_build
mkdir cmake_build
cd cmake_build

set +u

#cmake ${DIR} -B ${DIR}/cmake_build 
#make VERBOSE=1
#make install

cmake ${DIR} -B ${DIR}/cmake_build  -G Ninja -DCMAKE_BUILD_TYPE=Debug
ninja install

cd bin

ln -s libcudnn.so libcudnn.so.9
ln -s libcusparse.so libcusparse.so.12
ln -s libcuda.so libcuda.so.1
ln -s libcudart.so libcudart.so.12
ln -s libcufft.so libcufft.so.11
ln -s libcupti.so libcupti.so.12
ln -s libcublas.so libcublas.so.12
ln -s libcurand.so libcurand.so.10
ln -s libnccl.so libnccl.so.2
ln -s libnvidia-ml.so libnvidia-ml.so.1


#ln -s libhook_cublasLt.so libcublasLt.so.12

ls -lS 
set -u
