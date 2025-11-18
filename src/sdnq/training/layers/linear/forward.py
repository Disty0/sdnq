from typing import Tuple, Union

import torch
from ....common import compile_func, use_contiguous_mm

from ...tensor import SDNQTensor # noqa: TID252


def check_mats(input: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if input is not None:
        input = input.contiguous()
    if use_contiguous_mm:
        weight = weight.contiguous()
    elif weight.is_contiguous():
        weight = weight.t().contiguous().t()
    return input, weight


def linear_backward(
    grad_output: torch.FloatTensor,
    input: torch.FloatTensor,
    weight: torch.FloatTensor,
    bias: torch.FloatTensor = None,
    do_grad_input: bool = True,
    do_grad_weight: bool = True,
    do_grad_bias: bool = True,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    grad_output = grad_output.flatten(0,-2)
    if do_grad_input:
        grad_input = torch.mm(grad_output, weight).view(input.shape)
    if do_grad_weight:
        grad_weight = torch.mm(grad_output.t(), input.flatten(0,-2))
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class QuantizedLinearBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: Union[torch.FloatTensor, SDNQTensor], bias: torch.FloatTensor = None) -> torch.FloatTensor:
        if isinstance(weight, SDNQTensor):
            weight = weight.dequantize()
        ctx.save_for_backward(input, weight, bias)
        return torch.nn.functional.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias = ctx.saved_tensors
        return linear_backward(grad_output, input, weight, bias=bias, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return quantized_linear_with_backward(input, self.weight, self.bias)


quantized_linear_with_backward = QuantizedLinearBackward.apply
linear_backward = compile_func(linear_backward)
