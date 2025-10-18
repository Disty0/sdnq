from typing import Tuple, Union

import torch
from sdnq.common import compile_func

from ...dequantizer import SDNQTensor, quantize_fp8 # noqa: TID252
from .forward import quantized_linear_with_backward
from .linear_fp8_tensorwise import fp8_matmul_tensorwise
from .linear_fp8_tensorwise_dynamic import fp8_matmul_tensorwise_dynamic


def fp8_matmul_tensorwise_dynamic_ckpt(
    input: torch.FloatTensor,
    weight: torch.FloatTensor,
    bias: torch.FloatTensor = None,
    svd_up: torch.FloatTensor = None,
    svd_down: torch.FloatTensor = None,
    output_shape: torch.Size = None,
    do_input_reshape: bool = True,
) -> torch.FloatTensor:
    result = fp8_matmul_tensorwise_dynamic(input, weight, bias=bias, svd_up=svd_up, svd_down=svd_down)
    new_weight, weight_scale = quantize_fp8(weight, dim=0)
    new_input, input_scale = quantize_fp8(input.flatten(0,-2), dim=0)
    return result, new_input, new_weight, input_scale, weight_scale


def fp8_matmul_tensorwise_dynamic_backward_ckpt(
    grad_output: torch.FloatTensor,
    input: torch.FloatTensor,
    weight: torch.FloatTensor,
    input_scale: torch.FloatTensor,
    weight_scale: torch.FloatTensor,
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
        grad_input = fp8_matmul_tensorwise(grad_output, weight, weight_scale, svd_up=svd_up, svd_down=svd_down, output_shape=input_shape, do_input_reshape=False, use_sr=True)
    if do_grad_weight:
        grad_weight = fp8_matmul_tensorwise(grad_output.t(), input, input_scale, output_shape=None, do_input_reshape=False, use_sr=True)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class FP8MatmulTensorWiseDynamicBackwardCKPT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: Union[torch.FloatTensor, SDNQTensor], bias: torch.FloatTensor = None) -> torch.FloatTensor:
        svd_up, svd_down = None, None
        if isinstance(weight, SDNQTensor):
            svd_up, svd_down = weight.svd_up, weight.svd_down
            weight = weight.dequantize(non_svd=True)
        result, new_input, new_weight, input_scale, weight_scale = fp8_matmul_tensorwise_dynamic_ckpt_compiled(input, weight, bias=bias, svd_up=svd_up, svd_down=svd_down)
        ctx.save_for_backward(new_input, new_weight, input_scale, weight_scale, bias, svd_up, svd_down)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, input_scale, weight_scale, bias, svd_up, svd_down = ctx.saved_tensors
        return fp8_matmul_tensorwise_dynamic_backward_ckpt(grad_output, input, weight, input_scale, weight_scale, bias=bias, svd_up=svd_up, svd_down=svd_down, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward_fp8_matmul_tensorwise_dynamic_ckpt(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        if isinstance(self.weight, SDNQTensor):
            return quantized_linear_with_backward(input, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)
    return fp8_matmul_tensorwise_dynamic_with_backward_ckpt(input, self.weight, self.bias)


fp8_matmul_tensorwise_dynamic_with_backward_ckpt = FP8MatmulTensorWiseDynamicBackwardCKPT.apply
fp8_matmul_tensorwise_dynamic_ckpt_compiled = compile_func(fp8_matmul_tensorwise_dynamic_ckpt)
fp8_matmul_tensorwise_dynamic_backward_ckpt = compile_func(fp8_matmul_tensorwise_dynamic_backward_ckpt)
