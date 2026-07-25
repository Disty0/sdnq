import torch

from .....common import compile_func
from .....quant_utils import quantize_uint_mm, get_hadamard
from ....tensor import SDNQTensor

from ..forward import quantized_linear_with_backward
from .linear_uint8 import uint8_matmul
from .linear_uint8_dynamic import uint8_matmul_dynamic


def get_uint8_matmul_dynamic_backward_inputs(input: torch.FloatTensor, weight: torch.FloatTensor, hadamard: torch.FloatTensor | None, do_grad_weight: bool = True) -> tuple[torch.Tensor | None, torch.FloatTensor, torch.FloatTensor | None, torch.FloatTensor, torch.FloatTensor | None, torch.FloatTensor]:
    weight, scale, zero_point = quantize_uint_mm(weight.to(dtype=torch.float32), dim=0)
    if do_grad_weight:
        input, input_scale, input_zero_point = quantize_uint_mm(input.flatten(0,-2).to(dtype=torch.float32), dim=0, hadamard=hadamard)
        return input, weight, input_scale, scale, input_zero_point, zero_point
    return None, weight, None, scale, None, zero_point


def uint8_matmul_dynamic_backward_ckpt(
    grad_output: torch.FloatTensor,
    input: torch.FloatTensor,
    weight: torch.FloatTensor,
    input_scale: torch.FloatTensor,
    weight_scale: torch.FloatTensor,
    input_zero_point: torch.FloatTensor,
    weight_zero_point: torch.FloatTensor,
    bias: torch.FloatTensor | None = None,
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
        grad_input = uint8_matmul(
            grad_output, weight,
            weight_scale, weight_zero_point,
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            output_shape=output_shape,
            do_input_reshape=False,
            do_transpose=False,
        )
    if do_grad_weight:
        grad_weight = uint8_matmul(
            grad_output.t(), input,
            input_scale, input_zero_point,
            hadamard=hadamard,
            output_shape=None,
            do_input_reshape=False,
            do_transpose=False,
        )
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class UINT8MatmulDynamicBackwardCKPT(torch.autograd.Function):
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

        result = uint8_matmul_dynamic(
            input, weight,
            bias=bias,
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
        )

        new_input, new_weight, input_scale, weight_scale, input_zero_point, weight_zero_point = get_uint8_matmul_dynamic_backward_inputs(input, weight, hadamard, do_grad_weight=ctx.needs_input_grad[1])
        ctx.save_for_backward(new_input, new_weight, input_scale, weight_scale, input_zero_point, weight_zero_point, bias, svd_up, svd_down)
        ctx.input_shape = input.shape
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> tuple[torch.FloatTensor | None, torch.FloatTensor | None, torch.FloatTensor | None]:
        input, weight, input_scale, weight_scale, input_zero_point, weight_zero_point, bias, svd_up, svd_down = ctx.saved_tensors
        if ctx.use_hadamard:
            hadamard = get_hadamard(ctx.hadamard_group_size, dtype=grad_output.dtype, device=grad_output.device)
        else:
            hadamard = None

        return uint8_matmul_dynamic_backward_ckpt(
            grad_output, input, weight,
            input_scale, weight_scale,
            input_zero_point, weight_zero_point,
            bias=bias,
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            input_shape=ctx.input_shape,
            do_grad_input=ctx.needs_input_grad[0],
            do_grad_weight=ctx.needs_input_grad[1],
            do_grad_bias=ctx.needs_input_grad[2],
        )


def quantized_linear_forward_uint8_matmul_dynamic_ckpt(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        if isinstance(self.weight, SDNQTensor):
            return quantized_linear_with_backward(input, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)
    return uint8_matmul_dynamic_with_backward_ckpt(input, self.weight, self.bias)


uint8_matmul_dynamic_with_backward_ckpt = UINT8MatmulDynamicBackwardCKPT.apply
get_uint8_matmul_dynamic_backward_inputs = compile_func(get_uint8_matmul_dynamic_backward_inputs)
