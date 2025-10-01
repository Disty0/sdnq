from typing import Tuple

import torch
from sdnq.common import compile_func

from ...dequantizer import SDNQTensor, dequantize_symmetric, dequantize_symmetric_with_bias, quantize_fp8 # noqa: TID252
from .forward import check_mats, quantized_linear_with_backward


def quantize_fp8_matmul_tensorwise(input: torch.FloatTensor, weight: torch.FloatTensor, do_input_reshape: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2)
    else:
        weight = weight.t()
    weight, scale = quantize_fp8(weight, dim=-1)
    weight, scale = weight.t_(), scale.t_()
    input, input_scale = quantize_fp8(input, dim=-1)
    scale = torch.mul(input_scale, scale)
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, weight, scale


def fp8_matmul_tensorwise_dynamic(input: torch.FloatTensor, weight: torch.Tensor, bias: torch.FloatTensor, output_shape: torch.Size = None, do_input_reshape: bool = True) -> torch.FloatTensor:
    return_dtype = input.dtype
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[0] if do_input_reshape else weight.shape[-1]
    dummy_input_scale = torch.ones(1, device=input.device, dtype=torch.float32)
    input, weight, scale = quantize_fp8_matmul_tensorwise(input, weight, do_input_reshape=do_input_reshape)
    input, weight = check_mats(input, weight)
    if bias is not None:
        return dequantize_symmetric_with_bias(torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype), scale, bias, return_dtype, output_shape)
    else:
        return dequantize_symmetric(torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype), scale, return_dtype, output_shape)


def fp8_matmul_tensorwise_dynamic_backward(grad_output: torch.FloatTensor, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor, do_grad_input: bool = True, do_grad_weight: bool = True, do_grad_bias: bool = True) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    grad_output = grad_output.flatten(0,-2)
    if do_grad_input:
        grad_input = fp8_matmul_tensorwise_dynamic(grad_output, weight, None, output_shape=input.shape, do_input_reshape=False)
    if do_grad_weight:
        grad_weight = fp8_matmul_tensorwise_dynamic(grad_output.t(), input.flatten(0,-2), None, output_shape=None, do_input_reshape=False)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class FP8MatmulTensorWiseDynamicBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: torch.FloatTensor, bias: torch.FloatTensor) -> torch.FloatTensor:
        if isinstance(weight, SDNQTensor):
            weight = weight.dequantize()
        ctx.save_for_backward(input, weight, bias)
        return fp8_matmul_tensorwise_dynamic_compiled(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias = ctx.saved_tensors
        return fp8_matmul_tensorwise_dynamic_backward(grad_output, input, weight, bias, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward_fp8_matmul_tensorwise_dynamic(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        if isinstance(self.weight, SDNQTensor):
            return quantized_linear_with_backward(input, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)
    return fp8_matmul_tensorwise_dynamic_with_backward(input, self.weight, self.bias)


fp8_matmul_tensorwise_dynamic_with_backward = FP8MatmulTensorWiseDynamicBackward.apply
fp8_matmul_tensorwise_dynamic_compiled = compile_func(fp8_matmul_tensorwise_dynamic)
fp8_matmul_tensorwise_dynamic_backward = compile_func(fp8_matmul_tensorwise_dynamic_backward)
