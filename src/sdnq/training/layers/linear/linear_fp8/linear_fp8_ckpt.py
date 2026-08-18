import torch

from .....sdnext import devices
from .....common import compile_func
from .....dequantizer import dequantize_symmetric_compiled
from .....quant_utils import quantize_fp_mm, get_hadamard
from ....tensor import SDNQTensor

from ..forward import quantized_linear_with_backward
from .linear_fp8 import fp8_matmul
from .linear_fp8_dynamic import fp8_matmul_dynamic


@devices.inference_context()
def get_fp_matmul_backward_inputs(input: torch.FloatTensor, hadamard: torch.FloatTensor | None, matmul_dtype: str = "float8_e4m3fn") -> tuple[torch.Tensor, torch.FloatTensor]:
    input, input_scale = quantize_fp_mm(input.flatten(0,-2).to(dtype=torch.float32), dim=0, hadamard=hadamard, matmul_dtype=matmul_dtype)
    return input, input_scale


@devices.inference_context()
def fp8_matmul_backward_ckpt(
    grad_output: torch.FloatTensor,
    input: torch.FloatTensor | None,
    weight: torch.Tensor | None,
    input_scale: torch.FloatTensor | None,
    scale: torch.FloatTensor | None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    hadamard: torch.FloatTensor | None = None,
    input_shape: torch.Size | None = None,
    do_grad_input: bool = True,
    do_grad_weight: bool = True,
    do_grad_bias: bool = True,
) -> tuple[torch.FloatTensor | None, torch.FloatTensor | None, torch.FloatTensor | None]:
    grad_input = grad_weight = grad_bias = None
    output_shape = list(grad_output.shape)
    output_shape[-1] = input_shape[-1] if input_shape is not None else input.shape[-1]
    grad_output = grad_output.flatten(0,-2)
    if do_grad_input:
        grad_input = fp8_matmul_dynamic(
            grad_output,
            dequantize_symmetric_compiled(weight, scale),
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            output_shape=output_shape,
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
    if do_grad_bias:
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
            input,
            weight.weight,
            weight.scale,
            bias=bias,
            svd_up=weight.svd_up,
            svd_down=weight.svd_down,
            hadamard=hadamard,
            do_transpose=True,
        )

        if ctx.needs_input_grad[1]:
            new_input, input_scale = get_fp_matmul_backward_inputs(input, hadamard)
        else:
            new_input = input_scale = None
        ctx.input_shape = input.shape
        ctx.use_hadamard = weight.sdnq_dequantizer.use_hadamard
        ctx.hadamard_group_size = weight.sdnq_dequantizer.hadamard_group_size
        ctx.save_for_backward(new_input, weight if ctx.needs_input_grad[0] else None, input_scale)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> tuple[torch.FloatTensor | None, torch.FloatTensor | None, torch.FloatTensor | None]:
        input, weight, input_scale = ctx.saved_tensors
        if weight is not None:
            scale = weight.scale
            svd_up = weight.svd_up
            svd_down = weight.svd_down
            weight = weight.weight
        else:
            weight = scale = svd_up = svd_down = hadamard = None
        if ctx.use_hadamard and (weight is not None or input is not None):
            hadamard = get_hadamard(ctx.hadamard_group_size, dtype=grad_output.dtype, device=grad_output.device)
        else:
            hadamard = None

        return fp8_matmul_backward_ckpt(
            grad_output,
            input, weight,
            input_scale, scale,
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            input_shape=ctx.input_shape,
            do_grad_input=ctx.needs_input_grad[0],
            do_grad_weight=ctx.needs_input_grad[1],
            do_grad_bias=ctx.needs_input_grad[2],
        )


def quantized_linear_forward_fp8_matmul_ckpt(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return quantized_linear_with_backward(input, self.weight, self.bias)
    return fp8_matmul_with_backward_ckpt(input, self.weight, self.bias)


fp8_matmul_with_backward_ckpt = FP8MatmulBackwardCKPT.apply
get_fp_matmul_backward_inputs = compile_func(get_fp_matmul_backward_inputs)
