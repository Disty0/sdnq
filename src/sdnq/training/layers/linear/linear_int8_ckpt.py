from typing import Tuple

import torch
from ....common import compile_func

from ....dequantizer import dequantize_symmetric, quantize_int8
from .forward import quantized_linear_with_backward
from .linear_int8 import int8_matmul
from .linear_int8_dynamic import int8_matmul_dynamic
from ...tensor import SDNQTensor # noqa: TID252


def int8_matmul_ckpt(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    bias: torch.FloatTensor = None,
    svd_up: torch.FloatTensor = None,
    svd_down: torch.FloatTensor = None,
    output_shape: torch.Size = None,
    do_input_reshape: bool = True,
    do_transpose: bool = False,
) -> torch.FloatTensor:
    result = int8_matmul(input, weight, scale, bias=bias, svd_up=svd_up, svd_down=svd_down, output_shape=output_shape, do_input_reshape=do_input_reshape, do_transpose=do_transpose)
    input, input_scale = quantize_int8(input.flatten(0,-2), dim=0)
    return result, input, input_scale


def int8_matmul_backward_ckpt(
    grad_output: torch.FloatTensor,
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    input_scale: torch.FloatTensor,
    bias: torch.FloatTensor = None,
    svd_up: torch.FloatTensor = None,
    svd_down: torch.FloatTensor = None,
    do_grad_input: bool = True,
    do_grad_weight: bool = True,
    do_grad_bias: bool = True,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    input_shape = list(grad_output.shape)
    input_shape[-1] = input.shape[-1]
    grad_output = grad_output.flatten(0,-2)
    if do_grad_input:
        grad_input = int8_matmul_dynamic(grad_output, dequantize_symmetric(weight, scale), svd_up=svd_up, svd_down=svd_down, output_shape=input.shape, do_input_reshape=False, use_sr=True)
    if do_grad_weight:
        grad_weight = int8_matmul(grad_output.t(), input, input_scale, output_shape=None, do_input_reshape=False, is_backward_pass=True, use_sr=True)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class INT8MatmulBackwardCKPT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: SDNQTensor, bias: torch.FloatTensor = None) -> torch.FloatTensor:
        result, new_input, input_scale = int8_matmul_ckpt_compiled(input, weight.weight, weight.scale, bias=bias, svd_up=weight.svd_up, svd_down=weight.svd_down, do_transpose=True)
        ctx.save_for_backward(new_input, weight, input_scale, bias)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, input_scale, bias = ctx.saved_tensors
        return int8_matmul_backward_ckpt(grad_output, input, weight.weight, weight.scale, input_scale, bias=bias, svd_up=weight.svd_up, svd_down=weight.svd_down, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward_int8_matmul_ckpt(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return quantized_linear_with_backward(input, self.weight, self.bias)
    return int8_matmul_with_backward_ckpt(input, self.weight, self.bias)


int8_matmul_with_backward_ckpt = INT8MatmulBackwardCKPT.apply
int8_matmul_ckpt_compiled = compile_func(int8_matmul_ckpt)
int8_matmul_backward_ckpt = compile_func(int8_matmul_backward_ckpt)
