import torch

from .....quant_utils import get_hadamard
from ....tensor import SDNQTensor

from ..forward import quantized_linear_with_backward
from ..linear_fp8.linear_fp8_dynamic_ckpt import get_fp_matmul_dynamic_backward_inputs
from .linear_fp16 import fp16_matmul
from .linear_fp16_dynamic import fp16_matmul_dynamic


def fp16_matmul_dynamic_backward_ckpt(
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
        grad_input = fp16_matmul(
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
        grad_weight = fp16_matmul(
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


class FP16MatmulDynamicBackwardCKPT(torch.autograd.Function):
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

        result = fp16_matmul_dynamic(
            input, weight,
            bias=bias,
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
        )

        new_input, new_weight, input_scale, weight_scale = get_fp_matmul_dynamic_backward_inputs(
            input, weight,
            hadamard=hadamard,
            matmul_dtype="float16",
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

        return fp16_matmul_dynamic_backward_ckpt(
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


def quantized_linear_forward_fp16_matmul_dynamic_ckpt(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        if isinstance(self.weight, SDNQTensor):
            return quantized_linear_with_backward(input, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)
    return fp16_matmul_dynamic_with_backward_ckpt(input, self.weight, self.bias)


fp16_matmul_dynamic_with_backward_ckpt = FP16MatmulDynamicBackwardCKPT.apply
