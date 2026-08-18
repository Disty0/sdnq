import torch

from .....sdnext import devices
from .....common import compile_func
from .....quant_utils import quantize_fp_mm, get_hadamard
from ....tensor import SDNQTensor

from ..forward import quantized_linear_with_backward
from .linear_fp8 import fp8_matmul
from .linear_fp8_dynamic import fp8_matmul_dynamic


@devices.inference_context()
def get_fp_matmul_dynamic_backward_inputs(
    input: torch.FloatTensor,
    weight: torch.FloatTensor,
    hadamard: torch.FloatTensor | None = None,
    matmul_dtype: str = "float8_e4m3fn",
    do_grad_input: bool = True,
    do_grad_weight: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.FloatTensor | None, torch.FloatTensor | None]:
    if do_grad_input:
        weight, scale = quantize_fp_mm(weight.to(dtype=torch.float32), dim=0, matmul_dtype=matmul_dtype)
    else:
        weight = scale = None
    if do_grad_weight:
        input, input_scale = quantize_fp_mm(input.flatten(0,-2).to(dtype=torch.float32), dim=0, hadamard=hadamard, matmul_dtype=matmul_dtype)
    else:
        input = input_scale = None
    return input, weight, input_scale, scale


@devices.inference_context()
def fp8_matmul_dynamic_backward_ckpt(
    grad_output: torch.FloatTensor,
    input: torch.FloatTensor | None,
    weight: torch.FloatTensor | None,
    input_scale: torch.FloatTensor | None,
    weight_scale: torch.FloatTensor | None,
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
        grad_input = fp8_matmul(
            grad_output,
            weight, weight_scale,
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            output_shape=output_shape,
            do_input_reshape=False,
            do_transpose=False,
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


class FP8MatmulDynamicBackwardCKPT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: torch.FloatTensor | SDNQTensor, bias: torch.FloatTensor | None = None) -> torch.FloatTensor:
        if isinstance(weight, SDNQTensor):
            svd_up, svd_down = weight.svd_up, weight.svd_down
            ctx.use_hadamard = weight.sdnq_dequantizer.use_hadamard
            ctx.hadamard_group_size = weight.sdnq_dequantizer.hadamard_group_size
            weight = weight.dequantize(non_svd=True, non_hadamard=True)
        else:
            svd_up, svd_down = None, None
            ctx.use_hadamard = False
            ctx.hadamard_group_size = 256
        if ctx.use_hadamard:
            hadamard = get_hadamard(ctx.hadamard_group_size, dtype=input.dtype, device=input.device)
        else:
            hadamard = None

        result = fp8_matmul_dynamic(
            input, weight,
            bias=bias,
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
        )

        new_input, new_weight, input_scale, weight_scale = get_fp_matmul_dynamic_backward_inputs(
            input, weight,
            hadamard=hadamard,
            do_grad_input=ctx.needs_input_grad[0],
            do_grad_weight=ctx.needs_input_grad[1],
        )
        ctx.save_for_backward(new_input, new_weight, input_scale, weight_scale, svd_up, svd_down)
        ctx.input_shape = input.shape
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> tuple[torch.FloatTensor | None, torch.FloatTensor | None, torch.FloatTensor | None]:
        input, weight, input_scale, scale, svd_up, svd_down = ctx.saved_tensors
        if ctx.use_hadamard and (weight is not None or input is not None):
            hadamard = get_hadamard(ctx.hadamard_group_size, dtype=grad_output.dtype, device=grad_output.device)
        else:
            hadamard = None

        return fp8_matmul_dynamic_backward_ckpt(
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


def quantized_linear_forward_fp8_matmul_dynamic_ckpt(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        if isinstance(self.weight, SDNQTensor):
            return quantized_linear_with_backward(input, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)
    return fp8_matmul_dynamic_with_backward_ckpt(input, self.weight, self.bias)


fp8_matmul_dynamic_with_backward_ckpt = FP8MatmulDynamicBackwardCKPT.apply
get_fp_matmul_dynamic_backward_inputs = compile_func(get_fp_matmul_dynamic_backward_inputs)
