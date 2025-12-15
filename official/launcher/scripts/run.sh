# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
RUN_TYPE=test
NSYS_HW_INFO=True
# TYPE="Torch Megatron TE_unfused TE_fuse_softmax TE_flash-attn"
# SEQ="2048 4096 8192 16384"
CUDA_VISIBLE_DEVICES=4,5,6,7
TYPE="TE_flash-attn"
SEQ="16384"

if [ "${RUN_TYPE}" == "nsys" ]; then
    NSYS_EXEC="nsys profile --force-overwrite=true --trace=cuda,cudnn,cublas,cusparse,osrt,nvtx \
    --capture-range=cudaProfilerApi --capture-range-end=stop"
    if [ "${NSYS_HW_INFO}" == "True" ]; then
        NSYS_EXEC+=" -cuda-memory-usage true --gpu-metrics-device all"
    fi
else
    RUN_EXEC=torchrun
fi

DISTRIBUTED_ARGS="--nproc_per_node=4 --master_port=23456"

for layer_type in ${TYPE[@]}; do
    for seq_length in ${SEQ[@]}; do
        TEST_ID_RUN=${seq_length}_${layer_type}_shape
        if [ "${RUN_TYPE}" == "nsys" ]; then
            RUN_EXEC="${NSYS_EXEC} -o timeline_sdpa_${TEST_ID_RUN} torchrun"
        fi
        $RUN_EXEC $DISTRIBUTED_ARGS scheduler.py \
            --layer_type ${layer_type} --seq_list ${seq_length}
        echo "Done: ${TEST_ID_RUN}"
    done
done
