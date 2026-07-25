import torch
from ....common import compile_func
from ....kernel_wrappers import use_contiguous_int8_mm, use_contiguous_fp16_mm, use_contiguous_fp8_mm

from ...tensor import SDNQTensor


def check_mats(input: torch.Tensor, weight: torch.Tensor, matmul_dtype: str = "int8") -> tuple[torch.Tensor, torch.Tensor]:
    if input is not None:
        input = input.contiguous()
    if (
        (use_contiguous_int8_mm and matmul_dtype in {"int8", "uint8"})
        or (use_contiguous_fp16_mm and matmul_dtype in {"fp16", "float16"})
        or (use_contiguous_fp8_mm and matmul_dtype in {"fp8", "float8_e4m3fn"})
    ):
        weight = weight.contiguous()
    elif weight.is_contiguous():
        weight = weight.t().contiguous().t()
    return input, weight


def linear_backward(
    grad_output: torch.FloatTensor,
    input: torch.FloatTensor | None,
    weight: torch.FloatTensor | None,
    input_shape: torch.Size | None = None,
    do_grad_input: bool = True,
    do_grad_weight: bool = True,
    do_grad_bias: bool = True,
) -> tuple[torch.FloatTensor | None, torch.FloatTensor | None, torch.FloatTensor | None]:
    grad_input = grad_weight = grad_bias = None
    grad_output = grad_output.flatten(0,-2)
    if do_grad_input:
        grad_input = torch.mm(grad_output, weight).view(input_shape if input_shape is not None else input.shape)
    if do_grad_weight:
        grad_weight = torch.mm(grad_output.t(), input.flatten(0,-2))
    if do_grad_bias:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class QuantizedLinearBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: torch.FloatTensor | SDNQTensor, bias: torch.FloatTensor | None = None) -> torch.FloatTensor:
        ctx.input_shape = input.shape
        if isinstance(weight, SDNQTensor):
            weight = weight.dequantize()
        ctx.save_for_backward(input if ctx.needs_input_grad[1] else None, weight if ctx.needs_input_grad[0] else None)
        return torch.nn.functional.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> tuple[torch.FloatTensor | None, torch.FloatTensor | None, torch.FloatTensor | None]:
        input, weight = ctx.saved_tensors
        return linear_backward(
            grad_output,
            input, weight,
            input_shape=ctx.input_shape,
            do_grad_input=ctx.needs_input_grad[0],
            do_grad_weight=ctx.needs_input_grad[1],
            do_grad_bias=ctx.needs_input_grad[2],
        )


class QuantizedLinearBackwardCKPT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: torch.FloatTensor | SDNQTensor, bias: torch.FloatTensor | None = None) -> torch.FloatTensor:
        ctx.input_shape = input.shape
        ctx.save_for_backward(input if ctx.needs_input_grad[1] else None, weight if ctx.needs_input_grad[0] else None)
        if isinstance(weight, SDNQTensor):
            weight = weight.dequantize()
        return torch.nn.functional.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> tuple[torch.FloatTensor | None, torch.FloatTensor | None, torch.FloatTensor | None]:
        input, weight = ctx.saved_tensors
        if isinstance(weight, SDNQTensor):
            weight = weight.dequantize()
        return linear_backward(
            grad_output,
            input, weight,
            input_shape=ctx.input_shape,
            do_grad_input=ctx.needs_input_grad[0],
            do_grad_weight=ctx.needs_input_grad[1],
            do_grad_bias=ctx.needs_input_grad[2],
        )


def quantized_linear_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return quantized_linear_with_backward(input, self.weight, self.bias)


def quantized_linear_forward_ckpt(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return quantized_linear_with_backward_ckpt(input, self.weight, self.bias)


quantized_linear_with_backward = QuantizedLinearBackward.apply
quantized_linear_with_backward_ckpt = QuantizedLinearBackwardCKPT.apply
linear_backward = compile_func(linear_backward)
