from typing import Tuple, Union

import torch
from sdnq.common import compile_func, use_contiguous_mm

from sdnq.dequantizer import quantize_fp8, quantize_fp8_sr
from .forward import check_mats, quantized_linear_with_backward
from ...tensor import SDNQTensor # noqa: TID252


def quantize_fp8_matmul(input: torch.FloatTensor, weight: torch.FloatTensor, do_input_reshape: bool = True, use_sr: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2)
    else:
        weight = weight.t()
    weight, scale = quantize_fp8(weight.to(dtype=torch.float32), dim=-1)
    weight, scale = weight.t_(), scale.t_()
    if use_sr:
        input, input_scale = quantize_fp8_sr(input.to(dtype=torch.float32), dim=-1)
    else:
        input, input_scale = quantize_fp8(input.to(dtype=torch.float32), dim=-1)
    return input, weight, input_scale, scale


def fp8_matmul_dynamic(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    bias: torch.FloatTensor = None,
    svd_up: torch.FloatTensor = None,
    svd_down: torch.FloatTensor = None,
    output_shape: torch.Size = None,
    do_input_reshape: bool = True,
    use_sr: bool = False,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[0] if do_input_reshape else weight.shape[-1]
    if svd_up is not None:
        input = input.flatten(0,-2)
        svd_up, svd_down = svd_up.to(dtype=return_dtype), svd_down.to(dtype=return_dtype)
        if do_input_reshape:
            if use_contiguous_mm:
                svd_up, svd_down = svd_up.t().contiguous(), svd_down.t().contiguous()
            else:
                svd_up, svd_down = svd_up.contiguous().t(), svd_down.contiguous().t()
            svd_bias = torch.mm(torch.mm(input, svd_down), svd_up)
        else:
            _, svd_up = check_mats(None, svd_up)
            _, svd_down = check_mats(None, svd_down)
            svd_bias = torch.mm(torch.mm(input, svd_up), svd_down)
    input, weight, input_scale, scale = quantize_fp8_matmul(input, weight, do_input_reshape=do_input_reshape, use_sr=use_sr)
    input, weight = check_mats(input, weight)
    if bias is not None and bias.dtype != torch.bfloat16:
        bias = bias.to(dtype=torch.bfloat16)
    result = torch._scaled_mm(input, weight, scale_a=input_scale, scale_b=scale, bias=bias, out_dtype=torch.bfloat16).view(output_shape).to(return_dtype)
    if svd_up is not None:
        result = result.add_(svd_bias)
    return result


def fp8_matmul_dynamic_backward(
    grad_output: torch.FloatTensor,
    input: torch.FloatTensor,
    weight: torch.FloatTensor,
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
        grad_input = fp8_matmul_dynamic(grad_output, weight, svd_up=svd_up, svd_down=svd_down, output_shape=input.shape, do_input_reshape=False, use_sr=True)
    if do_grad_weight:
        grad_weight = fp8_matmul_dynamic(grad_output.t(), input.flatten(0,-2), output_shape=None, do_input_reshape=False, use_sr=True)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class FP8MatmulDynamicBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: Union[torch.FloatTensor, SDNQTensor], bias: torch.FloatTensor = None) -> torch.FloatTensor:
        svd_up, svd_down = None, None
        if isinstance(weight, SDNQTensor):
            svd_up, svd_down = weight.svd_up, weight.svd_down
            weight = weight.dequantize(non_svd=True)
        ctx.save_for_backward(input, weight, bias, svd_up, svd_down)
        return fp8_matmul_dynamic_compiled(input, weight, bias=bias, svd_up=svd_up, svd_down=svd_down)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias, svd_up, svd_down = ctx.saved_tensors
        return fp8_matmul_dynamic_backward(grad_output, input, weight, bias=bias, svd_up=svd_up, svd_down=svd_down, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


def quantized_linear_forward_fp8_matmul_dynamic(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        if isinstance(self.weight, SDNQTensor):
            return quantized_linear_with_backward(input, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)
    return fp8_matmul_dynamic_with_backward(input, self.weight, self.bias)


fp8_matmul_dynamic_with_backward = FP8MatmulDynamicBackward.apply
fp8_matmul_dynamic_compiled = compile_func(fp8_matmul_dynamic)
fp8_matmul_dynamic_backward = compile_func(fp8_matmul_dynamic_backward)
