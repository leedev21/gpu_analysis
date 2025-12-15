from importlib import import_module
import sys
import subprocess

def get_vllm_version():
    """Get vLLM version"""
    try:
        import vllm
        return vllm.__version__
    except ImportError:
        return None

def parse_version(version_str):
    """Parse version string to tuple for comparison"""
    if not version_str:
        return (0, 0, 0)
    
    # Remove any pre-release suffixes like .post1, .dev0, etc.
    version_str = version_str.split('.post')[0].split('.dev')[0].split('rc')[0]
    
    try:
        parts = version_str.split('.')
        return tuple(int(part) for part in parts[:3])
    except (ValueError, IndexError):
        return (0, 0, 0)

# vLLM 0.8.x package structure (updated based on detection results)
vllm_triton_op_v08 = {
    'vllm.lora.ops.triton_ops': {
        'bgmv_expand': '_bgmv_expand_kernel',
        'bgmv_shrink': '_bgmv_shrink_kernel',
        'sgmv_expand': '_sgmv_expand_kernel',
        'bgmv_expand_slice': '_bgmv_expand_slice_kernel',
        'sgmv_shrink': '_sgmv_shrink_kernel',
    },
    'vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm': [
        'scaled_mm_kernel',
    ],
    'vllm.model_executor.layers.quantization.awq_triton': [
        'awq_dequantize_kernel',
        'awq_gemm_kernel',
    ],
    'vllm.model_executor.layers.quantization.utils.fp8_utils': [
        '_per_token_group_quant_fp8',
        '_per_token_group_quant_fp8_colmajor',
        '_w8a8_block_fp8_matmul'
    ],
    'vllm.model_executor.layers.mamba.ops.mamba_ssm': [
        'softplus',
        '_selective_scan_update_kernel',
    ],
    'vllm.model_executor.layers.fused_moe.fused_moe': [
        'fused_moe',
        "fused_topk",
        "fused_experts",
        "get_config_file_name",
        "grouped_topk",
        "fused_moe_kernel_gptq_awq",
        "fused_moe_kernel",
        "moe_align_block_size_stage1",
        "moe_align_block_size_stage2",
        "moe_align_block_size_stage3",
        "moe_align_block_size_stage4",
    ],
    'vllm.attention.ops.triton_decode_attention': [
        'tanh',
        "_fwd_kernel_stage1",
        "_fwd_grouped_kernel_stage1",
        "_fwd_kernel_stage2",
    ],
    'vllm.attention.ops.triton_flash_attention': [
        'cdiv_fn',
        'max_fn',
        'dropout_offsets',
        'dropout_rng',
        'dropout_mask',
        'load_fn',
        '_attn_fwd_inner',
        'attn_fwd',
    ],
    'vllm.attention.ops.blocksparse_attention.blocksparse_attention_kernel': [
        '_fwd_kernel_inner',
        '_fwd_kernel_batch_inference'
    ],
    'vllm.attention.ops.prefix_prefill': [
        'context_attention_fwd',
        '_fwd_kernel',
        '_fwd_kernel_flash_attn_v2',
        '_fwd_kernel_alibi',
    ],
    'vllm.v1.attention.backends.flash_attn': [
        'merge_attn_states_kernel'
    ],
}

# vLLM 0.9.x package structure (updated based on the new structure)
vllm_triton_op_v09 = {
        'vllm.lora.ops.triton_ops.lora_expand_op': [
        '_lora_expand_kernel'
    ],
    
    'vllm.lora.ops.triton_ops.kernel_utils': [
        'mm_k',
        'do_expand_kernel',
        'do_shrink_kernel'
    ],
    'vllm.lora.ops.triton_ops.lora_shrink_op': [
        '_lora_shrink_kernel'
    ],
    'vllm.v1.spec_decode.utils': [
        'prepare_eagle_input_kernel'
    ],
    'vllm.v1.sample.rejection_sampler': [
        'rejection_greedy_sample_kernel',
        'rejection_random_sample_kernel',
        'expand_kernel',
        'sample_recovered_tokens_kernel'
    ],
    'vllm.model_executor.layers.mamba.ops.ssd_bmm': [
        '_bmm_chunk_fwd_kernel'
    ],
    'vllm.model_executor.layers.mamba.ops.mamba_ssm': [
        'softplus',
        '_selective_scan_update_kernel'
    ],
    'vllm.model_executor.layers.fused_moe.fused_batched_moe': [
        'moe_mmk',
        'expert_triton_kernel',
        'batched_triton_kernel'
    ],
    'vllm.model_executor.layers.lightning_attn': [
        '_fwd_diag_kernel',
        '_fwd_kv_parallel',
        '_fwd_kv_reduce',
        '_fwd_none_diag_kernel',
        '_linear_attn_decode_kernel'
    ],
    'vllm.model_executor.layers.quantization.awq_triton': [
        'awq_dequantize_kernel',
        'awq_gemm_kernel'
    ],
    'vllm.model_executor.layers.mamba.ops.ssd_chunk_state': [
        '_chunk_cumsum_fwd_kernel',
        '_chunk_state_fwd_kernel',
        '_chunk_state_varlen_kernel'
    ],
    'vllm.model_executor.layers.fused_moe.moe_align_block_size': [
        'moe_align_block_size_stage1',
        'moe_align_block_size_stage2',
        'moe_align_block_size_stage3',
        'moe_align_block_size_stage4'
    ],
    'vllm.model_executor.layers.quantization.utils.int8_utils': [
        'round_int8',
        '_per_token_quant_int8',
        '_per_token_group_quant_int8',
        '_w8a8_block_int8_matmul'
    ],
    'vllm.model_executor.layers.quantization.utils.fp8_utils': [
        '_per_token_group_quant_fp8',
        '_per_token_group_quant_fp8_colmajor',
        '_w8a8_block_fp8_matmul'
    ],
    'vllm.model_executor.layers.mamba.ops.ssd_chunk_scan': [
        '_chunk_scan_fwd_kernel'
    ],
    'vllm.model_executor.layers.mamba.ops.ssd_state_passing': [
        '_state_passing_fwd_kernel'
    ],
    'vllm.model_executor.layers.fused_moe.fused_moe': [
        'write_zeros_to_output',
        'fused_moe_kernel_gptq_awq',
        'fused_moe_kernel'
    ],
    'vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm': [
        'scaled_mm_kernel'
    ],
    'vllm.attention.ops.triton_unified_attention': [
        'cdiv_fn',
        'apply_softcap',
        'kernel_unified_attention_2d'
    ],
    'vllm.attention.ops.triton_decode_attention': [
        'tanh',
        '_fwd_kernel_stage1',
        '_fwd_grouped_kernel_stage1',
        '_fwd_kernel_stage2'
    ],
    'vllm.attention.ops.chunked_prefill_paged_decode': [
        'cdiv_fn',
        'kernel_paged_attention_2d'
    ],
    'vllm.attention.ops.triton_merge_attn_states': [
        'merge_attn_states_kernel'
    ],
    'vllm.attention.ops.triton_flash_attention': [
        'cdiv_fn',
        'max_fn',
        'dropout_offsets',
        'dropout_rng',
        'dropout_mask',
        'load_fn',
        '_attn_fwd_inner',
        'attn_fwd'
    ],
    'vllm.attention.ops.blocksparse_attention.blocksparse_attention_kernel': [
        '_fwd_kernel_inner',
        '_fwd_kernel_batch_inference'
    ],
    'vllm.attention.ops.prefix_prefill': [
        '_fwd_kernel',
        '_fwd_kernel_flash_attn_v2',
        '_fwd_kernel_alibi'
    ],
}

def get_vllm_triton_ops():
    """Get the appropriate vLLM triton ops mapping based on version"""
    version = get_vllm_version()
    version_tuple = parse_version(version)
    
    print(f"Detected vLLM version: {version}")
    
    # Version comparison: if version >= 0.9.0, use v09 mapping
    if version_tuple >= (0, 9, 0):
        print("Using vLLM 0.9+ package structure")
        return vllm_triton_op_v09
    else:
        print("Using vLLM 0.8.x package structure")
        return vllm_triton_op_v08

def get_vllm_path():
    """Find Transformer Engine install path using pip"""
    command = [sys.executable, "-m", "pip", "show", "vllm"]
    result = subprocess.run(command, capture_output=True, check=False, text=True)
    result = result.stdout.replace("\n", ":").split(":")
    if result and len(result) > 1:
        key = 'Editable project location' if 'Editable project location' in result else 'Location'
        return result[result.index(key) + 1].strip()
    else:
        return ""

def _load_vllm_triton():
    # Get the current version-appropriate mapping
    current_vllm_triton_op = get_vllm_triton_ops()
    
    if get_vllm_path():
        for op in current_vllm_triton_op:
            try:
                module = import_module(op)
                # print(module)
                for item in current_vllm_triton_op[op]:
                    try:
                        # print('\t', getattr(module, item))
                        yield module, item, getattr(module, item)
                    except AttributeError:
                        # Skip if the attribute doesn't exist in this version
                        print(f"Warning: {item} not found in {op}, skipping...")
                        continue
            except ImportError:
                # Skip if the module doesn't exist in this version
                print(f"Warning: Module {op} not found, skipping...")
                continue