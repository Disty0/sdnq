from typing import Tuple, Optional

import torch
from sdnq.common import compile_func, int_mm_func

from ...dequantizer import SDNQTensor, dequantize_symmetric, dequantize_symmetric_with_bias, quantize_int8, quantize_int8_sr # noqa: TID252
from .linear_int8_dynamic import int8_matmul_dynamic
from .forward import check_mats

try:
    from sdnq.triton_mm import int_mm as triton_int_mm
except ImportError:
    triton_int_mm = int_mm_func


def quantize_int8_matmul_input(input: torch.FloatTensor, scale: Optional[torch.FloatTensor] = None, dim: int = -1, do_input_reshape: bool = True, use_sr: bool = False) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2)
    if use_sr:
        input, input_scale = quantize_int8_sr(input, dim=dim)
    else:
        input, input_scale = quantize_int8(input, dim=dim)
    scale = torch.mul(input_scale, scale) if scale is not None else input_scale
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, scale


def int8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    bias: torch.FloatTensor = None,
    svd_up: torch.FloatTensor = None,
    svd_down: torch.FloatTensor = None,
    output_shape: torch.Size = None,
    do_input_reshape: bool = True,
    do_transpose: bool = False,
    is_backward_pass: bool = False,
    use_sr: bool = False,
) -> torch.FloatTensor:
    if is_backward_pass:
        int_mm = triton_int_mm if torch.version.cuda is not None and weight.device.type == "cuda" else int_mm_func
    else:
        int_mm = int_mm_func
    return_dtype = input.dtype
    if do_transpose:
        weight = weight.t()
        if scale is not None:
            scale = scale.t()
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[-1]
    if svd_up is not None:
        input = input.flatten(0,-2)
        svd_up, svd_down = svd_up.to(dtype=return_dtype), svd_down.to(dtype=return_dtype)
        if do_transpose:
            if bias is not None:
                bias = torch.addmm(bias, torch.mm(input, svd_down.t()), svd_up.t())
            else:
                bias = torch.mm(torch.mm(input, svd_down.t()), svd_up.t())
        else:
            if bias is not None:
                bias = torch.addmm(bias, torch.mm(input, svd_up), svd_down)
            else:
                bias = torch.mm(torch.mm(input, svd_up), svd_down)
    input, scale = quantize_int8_matmul_input(input, scale=scale, do_input_reshape=do_input_reshape, use_sr=use_sr)
    input, weight = check_mats(input, weight)
    if bias is not None:
        return dequantize_symmetric_with_bias(int_mm(input, weight), scale, bias, return_dtype, output_shape)
    else:
        return dequantize_symmetric(int_mm(input, weight), scale, return_dtype, output_shape)


def int8_matmul_backward(
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
        grad_input = int8_matmul_dynamic(grad_output, dequantize_symmetric(weight, scale), svd_up=svd_up, svd_down=svd_down, output_shape=input.shape, do_input_reshape=False, use_sr=True)
    if do_grad_weight:
        grad_weight = int8_matmul_dynamic(grad_output.t(), input.flatten(0,-2), output_shape=None, do_input_reshape=False, use_sr=True)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class INT8MatmulBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: SDNQTensor, bias: torch.FloatTensor = None) -> torch.FloatTensor:
        ctx.save_for_backward(input, weight, bias)
        return int8_matmul_compiled(input, weight.quant_data, weight.scale, bias=bias, svd_up=weight.svd_up, svd_down=weight.svd_down, do_transpose=True)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias = ctx.saved_tensors
        return int8_matmul_backward(grad_output, input, weight.quant_data, weight.scale, bias=bias, svd_up=weight.svd_up, svd_down=weight.svd_down, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward_int8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return quantized_linear_with_backward(input, self.weight, self.bias)
    return int8_matmul_with_backward(input, self.weight, self.bias)


int8_matmul_with_backward = INT8MatmulBackward.apply
int8_matmul_compiled = compile_func(int8_matmul)
int8_matmul_backward = compile_func(int8_matmul_backward)
