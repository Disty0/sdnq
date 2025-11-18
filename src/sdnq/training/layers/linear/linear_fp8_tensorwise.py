from typing import Tuple, Optional

import torch
from ....common import compile_func, use_contiguous_mm

from ....dequantizer import dequantize_symmetric, dequantize_symmetric_with_bias, quantize_fp8, quantize_fp8_sr
from .forward import check_mats, quantized_linear_with_backward
from .linear_fp8_tensorwise_dynamic import fp8_matmul_tensorwise_dynamic
from ...tensor import SDNQTensor # noqa: TID252


def quantize_fp8_matmul_input_tensorwise(input: torch.FloatTensor, scale: Optional[torch.FloatTensor] = None, dim: int = -1, do_input_reshape: bool = True, use_sr: bool = False) -> Tuple[torch.Tensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2)
    if use_sr:
        input, input_scale = quantize_fp8_sr(input.to(dtype=torch.float32), dim=dim)
    else:
        input, input_scale = quantize_fp8(input.to(dtype=torch.float32), dim=dim)
    scale = torch.mul(input_scale, scale) if scale is not None else input_scale
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, scale


def fp8_matmul_tensorwise(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    bias: torch.FloatTensor = None,
    svd_up: torch.FloatTensor = None,
    svd_down: torch.FloatTensor = None,
    output_shape: torch.Size = None,
    do_input_reshape: bool = True,
    do_transpose: bool = False,
    use_sr: bool = False,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    if do_transpose:
        weight = weight.t()
        scale = scale.t()
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[-1]
    if svd_up is not None:
        input = input.flatten(0,-2)
        svd_up, svd_down = svd_up.to(dtype=return_dtype), svd_down.to(dtype=return_dtype)
        if do_transpose:
            if use_contiguous_mm:
                svd_up, svd_down = svd_up.t().contiguous(), svd_down.t().contiguous()
            else:
                svd_up, svd_down = svd_up.contiguous().t(), svd_down.contiguous().t()
            if bias is not None:
                bias = torch.addmm(bias, torch.mm(input, svd_down), svd_up)
            else:
                bias = torch.mm(torch.mm(input, svd_down), svd_up)
        else:
            _, svd_up = check_mats(None, svd_up)
            _, svd_down = check_mats(None, svd_down)
            if bias is not None:
                bias = torch.addmm(bias, torch.mm(input, svd_up), svd_down)
            else:
                bias = torch.mm(torch.mm(input, svd_up), svd_down)
    dummy_input_scale = torch.ones(1, device=input.device, dtype=torch.float32)
    input, scale = quantize_fp8_matmul_input_tensorwise(input, scale=scale, do_input_reshape=do_input_reshape, use_sr=use_sr)
    input, weight = check_mats(input, weight)
    if bias is not None:
        return dequantize_symmetric_with_bias(torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype), scale, bias, dtype=return_dtype, result_shape=output_shape)
    else:
        return dequantize_symmetric(torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype), scale, dtype=return_dtype, result_shape=output_shape)


def fp8_matmul_tensorwise_backward(
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
        grad_input = fp8_matmul_tensorwise_dynamic(grad_output, dequantize_symmetric(weight, scale), svd_up=svd_up, svd_down=svd_down, output_shape=input.shape, do_input_reshape=False, use_sr=True)
    if do_grad_weight:
        grad_weight = fp8_matmul_tensorwise_dynamic(grad_output.t(), input.flatten(0,-2), output_shape=None, do_input_reshape=False, use_sr=True)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class FP8MatmulTensorWiseBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: SDNQTensor, bias: torch.FloatTensor = None) -> torch.FloatTensor:
        ctx.save_for_backward(input, weight, bias)
        return fp8_matmul_tensorwise_compiled(input, weight.weight, weight.scale, bias=bias, svd_up=weight.svd_up, svd_down=weight.svd_down, do_transpose=True)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias = ctx.saved_tensors
        return fp8_matmul_tensorwise_backward(grad_output, input, weight.weight, weight.scale, bias=bias, svd_up=weight.svd_up, svd_down=weight.svd_down, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward_fp8_matmul_tensorwise(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return quantized_linear_with_backward(input, self.weight, self.bias)
    return fp8_matmul_tensorwise_with_backward(input, self.weight, self.bias)


fp8_matmul_tensorwise_with_backward = FP8MatmulTensorWiseBackward.apply
fp8_matmul_tensorwise_compiled = compile_func(fp8_matmul_tensorwise)
fp8_matmul_tensorwise_backward = compile_func(fp8_matmul_tensorwise_backward)
