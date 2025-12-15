import os
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
from launcher.utils import (IS_TORCH,
                            all_modules,
                            trace_module,
                            search_modules,
                            prepare_input,
                            load_layer,
                            check_diff)

import megatron
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.datasets.utils import compile_helpers 
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.training.tokenizer.tokenizer import _NullTokenizer
from megatron.core import parallel_state
import torch.nn.functional as F
from .rms_norm import RMSNorm


DEVICE = None
MODEL_INIT = False

def initialize_distributed(args):
    global DEVICE
    tensor_model_parallel_size = args.training.tensor_model_parallel_size
    pipeline_model_parallel_size = args.training.pipeline_model_parallel_size
    context_parallel_size = args.training.context_parallel_size
    parallel_state.destroy_model_parallel()

    # Torch setup for distributed training
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    print('world_size:', world_size, 'rank:', rank)
    torch.distributed.init_process_group(backend='nccl', world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size,
                                             pipeline_model_parallel_size,
                                             context_parallel_size=context_parallel_size)
    DEVICE = torch.device("cuda")
    print('DEVICE:', DEVICE, 'tp:', tensor_model_parallel_size,
          'pp:', pipeline_model_parallel_size, 'cp:', context_parallel_size)
    return DEVICE, rank


def manual_seed(rank):
    if not IS_TORCH:
        model_parallel_cuda_manual_seed(123)


def model_provider(args):
    """Build the model."""
    global MODEL_INIT
    if not MODEL_INIT:
        megatron.core.models.gpt.gpt_layer_specs.LNImpl = RMSNorm
        megatron.core.transformer.transformer_block.LayerNormImpl = RMSNorm
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

        transformer_config = TransformerConfig(
            num_layers=args.model.num_layers,
            hidden_size=args.model.hidden_size,
            num_attention_heads=args.model.num_attention_heads,
            num_query_groups=args.model.num_query_groups if hasattr(args.model, 'num_query_groups') else None,
            ffn_hidden_size=args.model.ffn_hidden_size if hasattr(args.model, 'ffn_hidden_size') else None,
            normalization=args.model.normalization if hasattr(args.model, 'normalization') else 'LayerNorm',
            hidden_dropout=args.model.hidden_dropout if hasattr(args.model, 'hidden_dropout') else 0.1,
            attention_dropout=args.model.attention_dropout if hasattr(args.model, 'attention_dropout') else 0.1,
            gated_linear_unit=True if hasattr(args.model, 'swiglu') else False,
            activation_func=F.silu if hasattr(args.model, 'swiglu') else F.gelu,
            use_cpu_initialization=True,
            pipeline_dtype=torch.float32,
        )

        gpt_model = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size= args.model.vocab_size if hasattr(args.model, 'vocab_size') else 128,
            max_sequence_length=args.training.seq_length,
            position_embedding_type=args.model.position_embedding_type if hasattr(args.model, 'rotary_percent') else 'learned_absolute',
            rotary_percent=args.model.rotary_percent if hasattr(args.model, 'rotary_percent') else 1.0,
            rotary_base=args.model.rotary_base if hasattr(args.model, 'rotary_base') else 10000
        )
        MODEL_INIT = gpt_model.to(DEVICE)

    return MODEL_INIT


def get_train_data_iterator(args):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    config = GPTDatasetConfig(
        random_seed=0,
        sequence_length=args.training.seq_length,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=args.training.seq_length),
    )

    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, config
    ).build()

    train_dataloader = DataLoader(datasets[0], batch_size=args.training.micro_batch_size, shuffle=True)

    train_iterator = iter(train_dataloader)

    return train_iterator


def forward_step_func(data_iterator, model):

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        # If you have data parallel reduce loss across data parallel groups.
        # If pipeline parallel, loss computation is done only in last stage.

        return loss, {'lm loss': loss}

    data = next(data_iterator)
    tokens = data['tokens'].to(DEVICE)
    attention_mask = data['attention_mask'].to(DEVICE)
    position_ids = data['position_ids'].to(DEVICE)
    labels = data['labels'].to(DEVICE)
    loss_mask = data['loss_mask'].to(DEVICE)

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def splite_test_case(args):
    if hasattr(args.model, 'layers'):
        return args.model.layers
    return [None]


def prepare_test_case(test_case, args):
    print('prepare_test_case')
    gpt_model = model_provider(args)
    if not test_case:
        # all_modules(gpt_model, 'GPT', trace_module)
        optim = Adam(gpt_model.parameters())
        train_iterator = get_train_data_iterator(args)
        return {'model': (gpt_model, optim)}, train_iterator, get_forward_backward_func()
    else:
        layer_name, train_iterator = load_layer(test_case)
        layer = all_modules(gpt_model, 'GPT', search_modules(layer_name))
        print(layer_name, layer.__class__)
        return {'model': (layer, None)}, train_iterator, vanilla_loop


def forward_loop(module, batch, optim):
    args, kwargs = batch
    print('forward_loop')
    if kwargs:
        module(*args, **kwargs)
    else:
        module(*args)


def vanilla_loop(module, batch, optim):
    module_name = module.__class__.__name__
    rank = int(os.environ.get('LOCAL_RANK', 0))
    args, kwargs = batch['inputs']
    module.load_state_dict(batch['state_dict'])
    print('forward_loop:', module_name, rank, batch['init']['name'])
    # for i, a in enumerate(args):
    #     print(i, rank, str(a)[:50] + ' ,..., ' + str(a)[-50:])
    if kwargs:
        # for i, k in enumerate(kwargs):
        #     print(i, rank, k, str(kwargs[k])[:50] + ' ,..., ' + str(kwargs[k])[-50:])
        out = module(*args, **kwargs)
    else:
        out = module(*args)
    # print('out:', rank, str(out)[:50] + ' ,..., ' + str(out)[-50:])
    print('backward_loop:', module_name, rank, batch['init']['name'])
    if isinstance(out, tuple):
        out[0].backward(batch['in_bwd'])
    else:
        out.backward(batch['in_bwd'])

    if isinstance(out, (list, tuple)):
        return out
    else:
        return [out]


def run_iter(model_optim, data_iterator, forward_backward_func, num_steps, args):
    module, optim = model_optim
    if hasattr(args.model, 'layers'):
        num_steps = 1
        for _ in range(num_steps):
            data_iterator['inputs'] = prepare_input(DEVICE, data_iterator['inputs'])
            data_iterator['in_bwd'] = prepare_input(DEVICE, data_iterator['in_bwd'])
            res = forward_backward_func(module, data_iterator, optim)
            out = data_iterator['outputs'] if isinstance(data_iterator['outputs'], (tuple, list)) else (data_iterator['outputs'],)
            check_diff(res, out)
    else:
        for _ in range(num_steps):
            if not args.training.forward_only:
                optim.zero_grad()
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator,
                model=module,
                num_microbatches=args.training.gradient_accumulation_steps,
                seq_length=args.training.seq_length,
                micro_batch_size=args.training.micro_batch_size,
                decoder_seq_length=args.training.seq_length,
                forward_only=args.training.forward_only)
            if not args.training.forward_only:
                optim.step()
            print(f'Losses reduced :  {losses_reduced}')
