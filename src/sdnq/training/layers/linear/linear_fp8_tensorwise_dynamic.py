from typing import Tuple, Union

import torch

from ....common import compile_func, use_contiguous_mm
from ....dequantizer import dequantize_symmetric, dequantize_symmetric_with_bias, quantize_fp_mm, quantize_fp_mm_sr
from ...tensor import SDNQTensor # noqa: TID252

from .forward import check_mats, quantized_linear_with_backward


def quantize_fp_mm_tensorwise(input: torch.FloatTensor, weight: torch.FloatTensor, do_input_reshape: bool = True, use_sr: bool = False, matmul_dtype: str = "float8_e4m3fn") -> Tuple[torch.Tensor, torch.Tensor, torch.FloatTensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2)
    else:
        weight = weight.t()
    weight, scale = quantize_fp_mm(weight.to(dtype=torch.float32), dim=-1, matmul_dtype=matmul_dtype)
    weight, scale = weight.t_(), scale.t_()
    if use_sr:
        input, input_scale = quantize_fp_mm_sr(input.to(dtype=torch.float32), dim=-1, matmul_dtype=matmul_dtype)
    else:
        input, input_scale = quantize_fp_mm(input.to(dtype=torch.float32), dim=-1, matmul_dtype=matmul_dtype)
    scale = torch.mul(input_scale, scale)
    if scale.dtype == torch.float16: # fp16 will overflow
        scale = scale.to(dtype=torch.float32)
    return input, weight, scale


def fp8_matmul_tensorwise_dynamic(
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
    input, weight, scale = quantize_fp_mm_tensorwise(input, weight, do_input_reshape=do_input_reshape, use_sr=use_sr)
    input, weight = check_mats(input, weight)
    if bias is not None:
        return dequantize_symmetric_with_bias(torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype), scale, bias, dtype=return_dtype, result_shape=output_shape)
    else:
        return dequantize_symmetric(torch._scaled_mm(input, weight, scale_a=dummy_input_scale, scale_b=dummy_input_scale, bias=None, out_dtype=scale.dtype), scale, dtype=return_dtype, result_shape=output_shape)


def fp8_matmul_tensorwise_dynamic_backward(
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
        grad_input = fp8_matmul_tensorwise_dynamic(grad_output, weight, svd_up=svd_up, svd_down=svd_down, output_shape=input.shape, do_input_reshape=False)
    if do_grad_weight:
        grad_weight = fp8_matmul_tensorwise_dynamic(grad_output.t(), input.flatten(0,-2), output_shape=None, do_input_reshape=False)
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class FP8MatmulTensorWiseDynamicBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: Union[torch.FloatTensor, SDNQTensor], bias: torch.FloatTensor = None) -> torch.FloatTensor:
        svd_up, svd_down = None, None
        if isinstance(weight, SDNQTensor):
            svd_up, svd_down = weight.svd_up, weight.svd_down
            weight = weight.dequantize(non_svd=True)
        ctx.save_for_backward(input, weight, bias, svd_up, svd_down)
        return fp8_matmul_tensorwise_dynamic_compiled(input, weight, bias=bias, svd_up=svd_up, svd_down=svd_down)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias, svd_up, svd_down = ctx.saved_tensors
        return fp8_matmul_tensorwise_dynamic_backward(grad_output, input, weight, bias=bias, svd_up=svd_up, svd_down=svd_down, do_grad_input=ctx.needs_input_grad[0], do_grad_weight=ctx.needs_input_grad[1], do_grad_bias=ctx.needs_input_grad[2])


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
