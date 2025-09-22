from typing import Tuple

import torch
from sdnq.common import compile_func

from ...dequantizer import SDNQTensor, quantize_fp8 # noqa: TID252
from .forward import quantized_linear_with_backward # noqa: TID252


def check_fp8_mats(input: torch.Tensor, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    input_stride = input.stride()
    if not (input_stride[0] > input_stride[1] and input_stride[1] == 1):
        input = input.contiguous()
    weight_stride = weight.stride()
    if not (weight_stride[0] == 1 and weight_stride[1] > 1):
        weight = weight.t().contiguous().t()
    return input, weight


def quantize_fp8_matmul(input: torch.FloatTensor, weight: torch.FloatTensor, do_input_reshape: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2)
        weight = weight.t()
    weight, scale = quantize_fp8(weight, dim=0)
    input, input_scale = quantize_fp8(input, dim=-1)
    return input, weight, input_scale, scale


def fp8_matmul_dynamic(input: torch.FloatTensor, weight: torch.Tensor, bias: torch.FloatTensor, output_shape: torch.Size = None, do_input_reshape: bool = True) -> torch.FloatTensor:
    return_dtype = input.dtype
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[0] if do_input_reshape else weight.shape[-1]
    input, weight, input_scale, scale = quantize_fp8_matmul(input, weight, do_input_reshape=do_input_reshape)
    input, weight = check_fp8_mats(input, weight)
    if bias is not None and bias.dtype != torch.bfloat16:
        bias = bias.to(dtype=torch.bfloat16)
    return torch._scaled_mm(input, weight, scale_a=input_scale, scale_b=scale, bias=bias, out_dtype=torch.bfloat16).view(output_shape).to(return_dtype)


def fp8_matmul_dynamic_backward(grad_output: torch.FloatTensor, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor, do_grad_input: bool = True, do_grad_weight: bool = True, do_grad_bias: bool = True) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    grad_output = grad_output.flatten(0,-2)
    if do_grad_input:
        grad_input = fp8_matmul_dynamic(grad_output, weight, None, output_shape=input.shape, do_input_reshape=False)
    if do_grad_weight:
        grad_weight = fp8_matmul_dynamic(grad_output.t(), input.flatten(0,-2), None, output_shape=None, do_input_reshape=False)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class FP8MatmulBackwardDynamic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor) -> torch.FloatTensor:
        weight = weight.dequantize()
        ctx.save_for_backward(input, weight, bias)
        return fp8_matmul_dynamic_compiled(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias = ctx.saved_tensors
        return fp8_matmul_dynamic_backward(grad_output, input, weight, bias, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward_fp8_matmul_dynamic(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        if isinstance(self.weight, SDNQTensor):
            return quantized_linear_with_backward(input, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)
    return fp8_matmul_with_backward_dynamic(input, self.weight, self.bias)


fp8_matmul_with_backward_dynamic = FP8MatmulBackwardDynamic.apply
fp8_matmul_dynamic_compiled = compile_func(fp8_matmul_dynamic)
fp8_matmul_dynamic_backward = compile_func(fp8_matmul_dynamic_backward)
