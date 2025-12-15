# SPDX-License-Identifier: Apache-2.0
"""
Usage:
Single node:
    python examples/offline_inference/data_parallel.py \
            --model="ibm-research/PowerMoE-3b" \
            --dp-size=2 \
            --tp-size=2

Multi-node:
    Node 0 (assume the node has ip of 10.99.48.128):
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    --dp-size=2 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=0 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
    Node 1:
            python examples/offline_inference/data_parallel.py \
                    --model="ibm-research/PowerMoE-3b" \
                    --dp-size=2 \
                    --tp-size=2 \
                    --node-size=2 \
                    --node-rank=1 \
                    --master-addr=10.99.48.128 \
                    --master-port=13345
"""

import os
from time import sleep

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument(
        "--model",
        type=str,
        default="/devdata/data-llm/qwen/Qwen2.5-3B-Instruct-AWQ/",
        help="Model name or path",
    )
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--max-model-len", type=int, default=32768, help="Max model len")
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of prompts")
    parser.add_argument("--output-len", type=int, default=10, help="Output length")
    parser.add_argument("--num-hidden-layers", type=int, default=61)
    parser.add_argument(
        "--node-size", type=int, default=1, help="Total number of nodes"
    )
    parser.add_argument(
        "--node-rank", type=int, default=0, help="Rank of the current node"
    )
    parser.add_argument(
        "--master-addr", type=str, default="", help="Master node IP address"
    )
    parser.add_argument("--master-port", type=int, default=0, help="Master node port")
    parser.add_argument(
        "--enforce-eager", action="store_true", help="Enforce eager mode execution."
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code."
    )
    return parser.parse_args()


def main(
    model,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    GPUs_per_dp_rank,
    enforce_eager,
    trust_remote_code,
    max_model_len,
    num_prompts,
    output_len,
    num_hidden_layers,
):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # CUDA_VISIBLE_DEVICES for each DP rank is set automatically inside the
    # engine processes.

    # Sample prompts.
    prompts = [
        "Hello, my name is",
    ] * num_prompts

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    promts_per_rank = len(prompts) // dp_size
    start = global_dp_rank * promts_per_rank
    end = start + promts_per_rank
    prompts = prompts[start:end]
    if len(prompts) == 0:
        # if any rank has no prompts to process,
        # we need to set a placeholder prompt
        prompts = ["Placeholder"]
    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    # Create a sampling params object.
    # since we are doing data parallel, every rank can have different
    # sampling params. here we set different max_tokens for different
    # ranks for demonstration.
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=output_len,
    )

    # Create an LLM.
    llm = LLM(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=GPUs_per_dp_rank,
        enforce_eager=enforce_eager,
        enable_expert_parallel=True if dp_size > 1 else False,
        trust_remote_code=trust_remote_code,
        #hf_overrides={"num_hidden_layers": num_hidden_layers},
    )

    enable_torchtrace = os.getenv('VLLM_ENABLE_TORCHTRACE', '').strip().lower() == 'true'
    if enable_torchtrace:
        def rpc_func(worker):
            try:
                from torchtrace.torchtrace import update
            except:
                def update(*args, **kwargs):
                    pass
            print(worker.get_model())
            return update('model', 'qwen', worker.get_model())

        llm.llm_engine.engine_core.collective_rpc(rpc_func)

    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

    if enable_torchtrace:
        def rpc_func(worker):
            try:
                from torchtrace.torchtrace import set_torchtrace
            except:
                def set_torchtrace(*args, **kwargs):
                    pass
            set_torchtrace(torch_dispatch_trace=False, torch_api_trace=False)

        llm.llm_engine.engine_core.collective_rpc(rpc_func)

    # Print the outputs.
    for i, output in enumerate(outputs):
        if i >= 5:
            # print only 5 outputs
            break
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(
            f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
            f"Generated text: {generated_text!r}"
        )

    # Give engines time to pause their processing loops before exiting.
    sleep(1)


if __name__ == "__main__":
    args = parse_args()

    dp_size = args.dp_size
    tp_size = args.tp_size
    node_size = args.node_size
    node_rank = args.node_rank

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    from multiprocessing import Process

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)
    ):
        proc = Process(
            target=main,
            args=(
                args.model,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                tp_size,
                args.enforce_eager,
                args.trust_remote_code,
                args.max_model_len,
                args.num_prompts,
                args.output_len,
                args.num_hidden_layers,
            ),
        )
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=300)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)
