import torch

def all_gather():
    dim_size = list(input.size())
    dim_size[0] = dim_size[0] * tp_group.size()

    all_gather_buffer = get_global_memory_buffer().get_tensor(
        dim_size, input.dtype, "mpu"
    )
    handle = dist_all_gather_func(
        all_gather_buffer, input, group=tp_group, async_op=True
    )

    # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
    # gather is scheduled before the input gradient computation
    total_input = all_gather_buffer


def all_reduce():
    handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)


def reduce_scatter():
    dim_size = list(input.size())
    sub_grad_input = torch.empty(
        dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
    )
    # reduce_scatter
    handle = dist_reduce_scatter_func(
        sub_grad_input, grad_input, group=tp_group, async_op=True
    )
    # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
    # reduce scatter is scheduled before the weight gradient computation


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        """Forward function."""
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = group.size()
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )