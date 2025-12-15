#!/bin/bash
set -e

TORCHTRACE_PLUGIN_MODE=filter \
TORCHTRACE_PLUGIN_CONFIG="op_trace,{op: default, save_tensor: all}" \
VLLM_FLASH_ATTN_VERSION=2 \
TRITON_ALWAYS_COMPILE=1 \
VLLM_PLUGINS=torchtrace \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
VLLM_ENABLE_TORCHTRACE=true \
TORCHTRACE_PLUGIN_API=False \
TORCHTRACE_PLUGIN_SAVE_TO='/data/cyy/torchtrace/data/pt' \
TORCHTRACE_PLUGIN_SAVE_PT=True \
python test_vllm_0.9.py # \
#   --model /datasets/deepseek-r1/ \
#   --dp-size 8 \
#   --tp-size 1 \
#   --num-prompts 8 \
#   --max-model-len 32768 \
#   --output-len 2 \
#   --enforce-eager \
#   --trust-remote-code \
# 2>&1 | tee torchtrace.log