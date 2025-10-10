from typing import Tuple

import torch
from sdnq.common import compile_func

from ...dequantizer import SDNQTensor, dequantize_symmetric, quantize_fp8 # noqa: TID252
from .linear_fp8_dynamic import fp8_matmul_dynamic
from .forward import check_mats


def quantize_fp8_matmul_input(input: torch.FloatTensor, dim: int = -1, do_input_reshape: bool = True) -> Tuple[torch.Tensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2)
    input, input_scale = quantize_fp8(input, dim=dim)
    return input, input_scale


def fp8_matmul(
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
    return_dtype = input.dtype
    if do_transpose:
        weight = weight.t()
        scale = scale.t()
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[-1]
    if svd_up is not None:
        input = input.flatten(0,-2).to(dtype=torch.float32)
        if do_transpose:
            svd_bias = torch.mm(torch.mm(input, svd_down.t()), svd_up.t())
        else:
            svd_bias = torch.mm(torch.mm(input, svd_up), svd_down)
    input, input_scale = quantize_fp8_matmul_input(input, do_input_reshape=do_input_reshape)
    input, weight = check_mats(input, weight)
    if bias is not None and bias.dtype != torch.bfloat16:
        bias = bias.to(dtype=torch.bfloat16)
    result = torch._scaled_mm(input, weight, scale_a=input_scale, scale_b=scale, bias=bias, out_dtype=torch.bfloat16).view(output_shape).to(return_dtype)
    if svd_up is not None:
        result = result.add_(svd_bias)
    return result


def fp8_matmul_backward(
    grad_output: torch.FloatTensor,
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    bias: torch.FloatTensor = None,
    svd_up: torch.FloatTensor = None,
    svd_down: torch.FloatTensor = None,
    do_grad_input: bool = True,
    do_grad_weight: bool = True,
    do_grad_bias: bool = True,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    grad_output = grad_output.flatten(0,-2)
    if do_grad_input:
        weight = dequantize_symmetric(weight, scale)
        grad_input = fp8_matmul_dynamic(grad_output, weight, svd_up=svd_up, svd_down=svd_down, output_shape=input.shape, do_input_reshape=False)
    if do_grad_weight:
        grad_weight = fp8_matmul_dynamic(grad_output.t(), input.flatten(0,-2), output_shape=None, do_input_reshape=False)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class FP8MatmulBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: SDNQTensor, bias: torch.FloatTensor = None) -> torch.FloatTensor:
        ctx.save_for_backward(input, weight, bias)
        return fp8_matmul_compiled(input, weight.quant_data, weight.scale, bias=bias, svd_up=weight.svd_up, svd_down=weight.svd_down, do_transpose=True)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias = ctx.saved_tensors
        return fp8_matmul_backward(grad_output, input, weight.quant_data, weight.scale, bias=bias, svd_up=weight.svd_up, svd_down=weight.svd_down, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward_fp8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return quantized_linear_with_backward(input, self.weight, self.bias)
    return fp8_matmul_with_backward(input, self.weight, self.bias)


fp8_matmul_with_backward = FP8MatmulBackward.apply
fp8_matmul_compiled = compile_func(fp8_matmul)
fp8_matmul_backward = compile_func(fp8_matmul_backward)
