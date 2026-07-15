import torch

from .....common import compile_func
from .....dequantizer import dequantize_symmetric_compiled
from .....quant_utils import quantize_fp_mm, get_hadamard
from ....tensor import SDNQTensor

from ..forward import quantized_linear_with_backward
from .linear_fp8 import fp8_matmul
from .linear_fp8_dynamic import fp8_matmul_dynamic


def get_fp8_matmul_backward_inputs(input: torch.FloatTensor, hadamard: torch.FloatTensor | None) -> torch.FloatTensor:
    new_input, input_scale = quantize_fp_mm(input.flatten(0,-2).to(dtype=torch.float32), dim=0, hadamard=hadamard)
    return new_input, input_scale


def fp8_matmul_backward_ckpt(
    grad_output: torch.FloatTensor,
    input: torch.FloatTensor,
    weight: torch.Tensor,
    input_scale: torch.FloatTensor,
    scale: torch.FloatTensor,
    bias: torch.FloatTensor | None = None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    hadamard: torch.FloatTensor | None = None,
    do_grad_input: bool = True,
    do_grad_weight: bool = True,
    do_grad_bias: bool = True,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    input_shape = list(grad_output.shape)
    input_shape[-1] = input.shape[-1]
    grad_output = grad_output.flatten(0,-2)
    if do_grad_input:
        grad_input = fp8_matmul_dynamic(
            grad_output,
            dequantize_symmetric_compiled(weight, scale),
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            output_shape=input_shape,
            do_input_reshape=False,
        )
    if do_grad_weight:
        grad_weight = fp8_matmul(
            grad_output.t(),
            input, input_scale,
            hadamard=hadamard,
            output_shape=None,
            do_input_reshape=False,
            do_transpose=False,
        )
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class FP8MatmulBackwardCKPT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: SDNQTensor, bias: torch.FloatTensor | None = None) -> torch.FloatTensor:
        if weight.sdnq_dequantizer.use_hadamard:
            hadamard = get_hadamard(weight.sdnq_dequantizer.hadamard_group_size, dtype=input.dtype, device=input.device)
        else:
            hadamard = None

        result = fp8_matmul(
            input, weight.weight, weight.scale,
            bias=bias,
            svd_up=weight.svd_up,
            svd_down=weight.svd_down,
            hadamard=hadamard,
            do_transpose=True,
        )

        new_input, input_scale = get_fp8_matmul_backward_inputs(input, hadamard)
        ctx.save_for_backward(new_input, weight, input_scale, bias)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, input_scale, bias = ctx.saved_tensors
        if weight.sdnq_dequantizer.use_hadamard:
            hadamard = get_hadamard(weight.sdnq_dequantizer.hadamard_group_size, dtype=grad_output.dtype, device=grad_output.device)
        else:
            hadamard = None

        return fp8_matmul_backward_ckpt(
            grad_output, input, weight.weight,
            input_scale, weight.scale,
            bias=bias,
            svd_up=weight.svd_up,
            svd_down=weight.svd_down,
            hadamard=hadamard,
            do_grad_input=ctx.needs_input_grad[0],
            do_grad_weight=ctx.needs_input_grad[1],
            do_grad_bias=ctx.needs_input_grad[2],
        )


def quantized_linear_forward_fp8_matmul_ckpt(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return quantized_linear_with_backward(input, self.weight, self.bias)
    return fp8_matmul_with_backward_ckpt(input, self.weight, self.bias)


fp8_matmul_with_backward_ckpt = FP8MatmulBackwardCKPT.apply
get_fp8_matmul_backward_inputs = compile_func(get_fp8_matmul_backward_inputs)
