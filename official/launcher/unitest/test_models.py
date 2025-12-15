# SPDX-License-Identifier: Apache-2.0
"""Compare the outputs of HF and vLLM when using greedy sampling.

Run `pytest tests/models/test_models.py`.
"""
import pytest
import os
# try:
from torchtrace.torchtrace import set_torchtrace, update
# except:
#     set_torchtrace = None

# from ...utils import check_logprobs_close
try:
    import vllm._custom_ops as ops
except ImportError:
    print('vllm not found')
    ops = None


from conftest import *


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            "bigscience/bloom-560m",  # bloom - testing alibi slopes
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "openai-community/gpt2",  # gpt2
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param("Milos/slovak-gpt-j-405M"),  # gptj
        pytest.param("bigcode/tiny_starcoder_py"),  # gpt_bigcode
        pytest.param("EleutherAI/pythia-70m"),  # gpt_neox
        pytest.param(
            "google/gemma-1.1-2b-it",  # gemma
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "THUDM/chatglm3-6b",  # chatglm (text-only)
        ),
        pytest.param(
            "meta-llama/Llama-3.2-1B-Instruct",  # llama
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "openbmb/MiniCPM3-4B",
            # fused_moe not supported on CPU
            marks=[pytest.mark.core_model],
        ),
        pytest.param(
            "facebook/opt-125m",  # opt
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "microsoft/phi-2",  # phi
            marks=[pytest.mark.core_model],
        ),
        pytest.param(
            "Qwen/Qwen-7B",  # qwen (text-only)
        ),
        pytest.param(
            "Qwen/Qwen2.5-0.5B-Instruct",  # qwen2
            marks=[pytest.mark.core_model],
        ),
        pytest.param("stabilityai/stablelm-3b-4e1t"),  # stablelm
        pytest.param("bigcode/starcoder2-3b"),  # starcoder2
        pytest.param(
            "ehristoforu/Falcon3-MoE-2x7B-Insruct",  # mixtral
            marks=[pytest.mark.cpu_model],
        )
    ])


@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype,
    max_tokens: int,
    num_logprobs: int,
) -> None:

    # with hf_runner(model, dtype=dtype) as hf_model:
    #     if model.startswith("THUDM/chatglm3"):
    #         hf_model.model.get_output_embeddings = lambda: \
    #             hf_model.model.transformer.output_layer

    #     hf_outputs = hf_model.generate_greedy_logprobs_limit(
    #         example_prompts, max_tokens, num_logprobs)
    print('start')

    with vllm_runner(model, dtype=dtype, enforce_eager=True, quantization='fp8',) as vllm_model:
        # print('model:', vllm_model.model.llm_engine.model_executor.execute_model)
        model_config = vllm_model.model.llm_engine.model_config
        for _str in model.split('/'):
            if _str:
                model_name = _str
        for k, v in vllm_model.model.llm_engine.__dict__.items():
            print('--------------------------------------')
            print('k:', k)
            print(v)
            if k not in ['vllm_config', 'tokenizer', 'detokenizer', 'generation_config_fields']:
                try:
                    torch.save(v, os.path.join('data/pt', f'vllm_{model_name}_init_{k}.pt'))
                except:
                    print('save failed:', k)

        if os.getenv('VLLM_USE_V1') == '0':
            def register_model(model):
                # def calculate_tensor(trace_str, _tensor):
                #     _min, _max, _bins = -151, 128, 280
                #     positive = torch.histc(torch.log2(_tensor.to(torch.float)), min=_min, max=_max, bins=_bins)
                #     negative = torch.histc(torch.log2(-_tensor.to(torch.float)), min=_min, max=_max, bins=_bins)
                #     return {'name': trace_str, 'd': torch.stack((positive, negative), dim=0)}

                # for k, _tensor in model.state_dict().items():
                #     t = calculate_tensor('param_' + k, _tensor)
                #     torch.save(t, os.path.join('data/pt', f'param_{k}.pt'))
                print(model)
                try:
                    from torchtrace.torchtrace import update
                except:
                    def update(*args, **kwargs):
                        pass
                update('model', 'Qwen2.5', model)

            vllm_model.apply_model(register_model)
        else:
            def rpc_func(worker):
                try:
                    from torchtrace.torchtrace import update
                except:
                    def update(*args, **kwargs):
                        pass
                print(worker.get_model())
                return update('model', 'qwen', worker.get_model())
            vllm_model.model.llm_engine.engine_core.collective_rpc(rpc_func)

        print('********************')
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)
        print('********************\n')
        if isinstance(vllm_outputs, list):
            for line in vllm_outputs:
                print(line)
        else:
            print(vllm_outputs)

        # This test is for verifying whether the model's extra_repr
        # can be printed correctly.

    # check_logprobs_close(
    #     outputs_0_lst=hf_outputs,
    #     outputs_1_lst=vllm_outputs,
    #     name_0="hf",
    #     name_1="vllm",
    # )


# V1 engine(after 0.9): VLLM_ALLOW_INSECURE_SERIALIZATION=1 CUDA_VISIBLE_DEVICES=0 python launcher/unitest/test_models.py
# V0 engine(before 0.9): VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0 python launcher/unitest/test_models.py

if __name__ == "__main__":
    if set_torchtrace and os.getenv('RUN_TYPE', '') == 'trace':
        # update('customer_op')
        set_torchtrace(torch_dispatch_trace=True, torch_api_trace=True, save_pt=False, sync_mode=True)

    # test_models(None, VllmRunner, "Translate the following content into Chinese directly: DeepSeek-V2 adopts innovative architectures to guarantee economical training and efficient inference.", "/devdata/data-llm/qwen/Qwen2.5-3B-Instruct-AWQ/", torch.bfloat16, 5, 5)
    test_models(None, VllmRunner, ["Translate the following content into Chinese directly: DeepSeek-V2 adopts innovative architectures to guarantee economical training and efficient inference.",
                                   "How many numbers between 1 and 2005 are integer multiples of 3 or 4 but not 12?"], "/devdata/data-llm/deepseek/deepseek-r1-6-layers/", torch.bfloat16, 3, 4)