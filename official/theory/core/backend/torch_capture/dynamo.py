# see also: https://github.com/llvm/torch-mlir/blob/main/python/torch_mlir/dynamo.py

from __future__ import annotations
from typing import List, Callable
import warnings

import torch
from torch._functorch.compile_utils import strip_overloads
from torch._decomp import core_aten_decompositions
from torch._dynamo.backends.common import aot_autograd
import functorch

# https://github.com/pytorch/pytorch/issues/89064
warnings.filterwarnings("ignore", module="torch.jit._check")

def _adjust_calling_convention(gm: torch.fx.GraphModule) -> bool:
    """Canonicalize the calling convention to the one that have either
    a None return value, a single return value or a non-singleton tuple of
    return values. But various situations create functions with single-element
    tuples, or lists instead of tuples. This function adjusts the calling
    conventions to match, and returns the information needed for the calling
    code to reconstruct the original calling convention.

    Returns:
        Two booleans
        - The first indicates if a single-element tuple/list return
          was converted to a return of the element itself.
        - The second indicates if a list return was converted to a tuple.
    """
    did_unwrap_single_element = False
    did_convert_list_to_tuple = False
    for node in gm.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, \
                "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                if len(node_arg) == 1:
                    node.args = (node_arg[0],)
                    did_unwrap_single_element = True
                    break
            if isinstance(node_arg, list):
                if len(node_arg) == 1:
                    node.args = (node_arg[0],)
                    did_unwrap_single_element = True
                    did_convert_list_to_tuple = True
                    break
                node.args= (tuple(node_arg),)
                did_convert_list_to_tuple = True
                break

    if did_unwrap_single_element:
        gm.graph.lint()
        gm.recompile()
    return did_unwrap_single_element, did_convert_list_to_tuple


def make_simple_dynamo_backend(
    user_backend = None, *,
    decompositions : Callable = core_aten_decompositions,
):
    """Wrapper for functions intended to be used as TorchDynamo backends.

    Args:
        user_backend: A function with the signature used by ordinary TorchDynamo backends.
        decompositions: A function that returns the list of ops that will be decomposed.
            Default is the core_aten_decompositions function.
            See also: https://dev-discuss.pytorch.org/t/defining-the-core-aten-opset/1464

    Returns:
        A function with the signature used by TorchDynamo backends.
    """

    if user_backend is None:
        def fn(user_backend):
            assert user_backend, "user_backend must be provided"
            return make_simple_dynamo_backend(user_backend,
                decompositions=decompositions)
        return fn

    def wrapper_backend(gm: torch.fx.GraphModule,
                        example_inputs: List[torch.Tensor]):
        did_unwrap_single_element, did_convert_list_to_tuple = _adjust_calling_convention(gm)
        strip_overloads(gm)
        user_callable = user_backend(gm, example_inputs)

        # TODO: Have a consistent story about the boxed calling convention.
        # (for more details on this remove this decorator and look at the warning)
        # See https://github.com/pytorch/pytorch/pull/83137#issuecomment-1211320670 for rationale.
        @functorch.compile.make_boxed_func
        def dynamo_callable(*inputs):
            result = user_callable(*inputs)
            if did_unwrap_single_element:
                result = (result,)
            if did_convert_list_to_tuple:
                result = list(result)
            return result
        return dynamo_callable
    return aot_autograd(fw_compiler=wrapper_backend,
                        decompositions=decompositions())
