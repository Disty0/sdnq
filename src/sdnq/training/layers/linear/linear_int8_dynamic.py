from typing import Tuple

import torch
from sdnq.common import compile_func, int_mm_func

from ...dequantizer import SDNQTensor, dequantize_symmetric, dequantize_symmetric_with_bias, quantize_int8 # noqa: TID252
from .forward import check_mats, quantized_linear_with_backward

try:
    from sdnq.triton_mm import int_mm as triton_int_mm
except ImportError:
    triton_int_mm = int_mm_func

def quantize_int8_matmul(input: torch.FloatTensor, weight: torch.FloatTensor, do_input_reshape: bool = True) -> Tuple[torch.CharTensor, torch.CharTensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2)
        weight = weight.t()
    weight, scale = quantize_int8(weight, dim=0)
    input, input_scale = quantize_int8(input, dim=-1)
    scale = torch.mul(input_scale, scale)
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, weight, scale


def int8_matmul_dynamic(input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor, output_shape: torch.Size = None, do_input_reshape: bool = True) -> torch.FloatTensor:
    int_mm = triton_int_mm if torch.version.cuda is not None and weight.device.type == "cuda" else int_mm_func
    return_dtype = input.dtype
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[0] if do_input_reshape else weight.shape[-1]
    input, weight, scale = quantize_int8_matmul(input, weight, do_input_reshape=do_input_reshape)
    input, weight = check_mats(input, weight)
    if bias is not None:
        return dequantize_symmetric_with_bias(int_mm(input, weight), scale, bias, return_dtype, output_shape)
    else:
        return dequantize_symmetric(int_mm(input, weight), scale, return_dtype, output_shape)


def int8_matmul_dynamic_backward(grad_output: torch.FloatTensor, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor, do_grad_input: bool = True, do_grad_weight: bool = True, do_grad_bias: bool = True) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    grad_output = grad_output.flatten(0,-2)
    if do_grad_input:
        grad_input = int8_matmul_dynamic(grad_output, weight, None, output_shape=input.shape, do_input_reshape=False)
    if do_grad_weight:
        grad_weight = int8_matmul_dynamic(grad_output.t(), input.flatten(0,-2), None, output_shape=None, do_input_reshape=False)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class INT8MatmulDynamicBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor) -> torch.FloatTensor:
        if isinstance(weight, SDNQTensor):
            weight = weight.dequantize()
        ctx.save_for_backward(input, weight, bias)
        return int8_matmul_dynamic_compiled(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias = ctx.saved_tensors
        return int8_matmul_dynamic_backward(grad_output, input, weight, bias, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward_int8_matmul_dynamic(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        if isinstance(self.weight, SDNQTensor):
            return quantized_linear_with_backward(input, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)
    return int8_matmul_dynamic_with_backward(input, self.weight, self.bias)


int8_matmul_dynamic_with_backward = INT8MatmulDynamicBackward.apply
int8_matmul_dynamic_compiled = compile_func(int8_matmul_dynamic)
int8_matmul_dynamic_backward = compile_func(int8_matmul_dynamic_backward)
