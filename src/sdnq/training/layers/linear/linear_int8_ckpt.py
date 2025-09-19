from typing import Tuple

import torch
from sdnq.common import compile_func

from ...dequantizer import dequantize_symmetric
from .linear_int8 import int8_matmul, quantize_int8_matmul_input # noqa: TID252
from .linear_int8_dynamic import int8_matmul_dynamic # noqa: TID252


def int8_matmul_ckpt(input: torch.FloatTensor, weight: torch.Tensor, bias: torch.FloatTensor, scale: torch.FloatTensor, output_shape: torch.Size = None, do_input_reshape: bool = True, do_transpose: bool = False) -> torch.FloatTensor:
    result = int8_matmul(input, weight, bias, scale, output_shape=output_shape, do_input_reshape=do_input_reshape, do_transpose=do_transpose)
    new_input, input_scale = quantize_int8_matmul_input(input, dim=0)
    return result, new_input, input_scale


def int8_matmul_backward_ckpt(grad_output: torch.FloatTensor, input: torch.FloatTensor, weight: torch.Tensor, scale: torch.FloatTensor, bias: torch.FloatTensor, input_scale: torch.FloatTensor, do_grad_input: bool = True, do_grad_weight: bool = True, do_grad_bias: bool = True) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    input_shape = list(grad_output.shape)
    input_shape[-1] = input.shape[-1]
    grad_output = grad_output.flatten(0,-2)
    if do_grad_input:
        grad_input = int8_matmul_dynamic(grad_output, dequantize_symmetric(weight, scale), None, output_shape=input.shape, do_input_reshape=False)
    if do_grad_weight:
        grad_weight = int8_matmul(grad_output.t(), input, None, input_scale, output_shape=None, do_input_reshape=False)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class INT8MatmulBackwardCKPT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor) -> torch.FloatTensor:
        result, new_input, input_scale = int8_matmul_ckpt_compiled(input, weight.quant_data, bias, weight.scale, do_transpose=True)
        ctx.save_for_backward(new_input, weight, bias, input_scale)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias, input_scale = ctx.saved_tensors
        return int8_matmul_backward_ckpt(grad_output, input, weight.quant_data, weight.scale, bias, input_scale, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward_int8_matmul_ckpt(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return int8_matmul_with_backward_ckpt(input, self.weight, self.bias)


int8_matmul_with_backward_ckpt = INT8MatmulBackwardCKPT.apply
int8_matmul_ckpt_compiled = compile_func(int8_matmul_ckpt)
int8_matmul_backward_ckpt = compile_func(int8_matmul_backward_ckpt)
