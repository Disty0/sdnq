import torch

from .....common import compile_func
from .....kernel_wrappers import int_scaled_mm_func, use_contiguous_fp16_mm
from .....dequantizer import dequantize_symmetric_compiled, dequantize_asymmetric_compiled
from .....quant_utils import quantize_int_mm, rotate_hadamard, rotate_hadamard_compiled, get_hadamard
from ....tensor import SDNQTensor

from ..forward import check_mats, quantized_linear_with_backward
from .linear_int8_dynamic import int8_matmul_dynamic


def quantize_int_mm_input(input: torch.FloatTensor, dim: int = -1, do_input_reshape: bool = True, use_sr: bool = False) -> tuple[torch.Tensor, torch.FloatTensor]:
    if do_input_reshape:
        input = input.flatten(0,-2)
    input, input_scale = quantize_int_mm(input.to(dtype=torch.float32), dim=dim, use_sr=use_sr)
    return input, input_scale


def get_int8_matmul_inputs(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    bias: torch.FloatTensor | None = None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    zero_point: torch.FloatTensor | None = None,
    hadamard: torch.FloatTensor | None = None,
    output_shape: torch.Size = None,
    do_input_reshape: bool = True,
    do_transpose: bool = True,
    use_sr: bool = False,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    bias_to_add_after = None

    if do_transpose:
        weight = weight.t()
        scale = scale.t()
        if zero_point is not None:
            zero_point = zero_point.t()
    if weight.dtype == torch.uint8:
        weight = weight.bitwise_xor(128).view(torch.int8)
        if zero_point is not None:
            zero_point = torch.add(zero_point, scale, alpha=128)
        else:
            zero_point = torch.mul(scale, 128)

    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[-1]

    if hadamard is not None:
        if do_transpose:
            input = rotate_hadamard(input, hadamard=hadamard)
        else:
            bias_to_add_after = bias
            bias = None
    if svd_up is not None:
        input = input.flatten(0,-2)
        svd_up, svd_down = svd_up.to(dtype=return_dtype), svd_down.to(dtype=return_dtype)
        if do_transpose:
            if use_contiguous_fp16_mm:
                svd_up, svd_down = svd_up.t().contiguous(), svd_down.t().contiguous()
            else:
                svd_up, svd_down = svd_up.contiguous().t(), svd_down.contiguous().t()
            if bias is not None:
                bias = torch.addmm(bias, torch.mm(input, svd_down), svd_up)
            else:
                bias = torch.mm(torch.mm(input, svd_down), svd_up)
        else:
            _, svd_up = check_mats(None, svd_up, matmul_dtype="float16")
            _, svd_down = check_mats(None, svd_down, matmul_dtype="float16")
            if bias is not None:
                bias = torch.addmm(bias, torch.mm(input, svd_up), svd_down)
            else:
                bias = torch.mm(torch.mm(input, svd_up), svd_down)

    input, input_scale = quantize_int_mm_input(input, do_input_reshape=do_input_reshape, use_sr=use_sr)
    if zero_point is not None:
        zero_bias = torch.sum(input, dim=-1, keepdim=True, dtype=torch.int32).to(dtype=input_scale.dtype).mul_(input_scale).mul(zero_point)
        if bias is not None:
            zero_bias.add_(bias)
        bias = zero_bias
    input, weight = check_mats(input, weight, matmul_dtype="int8")
    return input, weight, input_scale, scale, bias, bias_to_add_after, return_dtype, output_shape


def int8_matmul(
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    bias: torch.FloatTensor | None = None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    zero_point: torch.FloatTensor | None = None,
    hadamard: torch.FloatTensor | None = None,
    output_shape: torch.Size = None,
    do_input_reshape: bool = True,
    do_transpose: bool = True,
    use_sr: bool = False,
) -> torch.FloatTensor:
    input, weight, input_scale, scale, bias, bias_to_add_after, return_dtype, output_shape = get_int8_matmul_inputs(
        input=input,
        weight=weight,
        scale=scale,
        bias=bias,
        svd_up=svd_up,
        svd_down=svd_down,
        zero_point=zero_point,
        hadamard=hadamard,
        output_shape=output_shape,
        do_input_reshape=do_input_reshape,
        do_transpose=do_transpose,
        use_sr=use_sr,
    )
    result = int_scaled_mm_func(input, weight, input_scale, scale, bias=bias, out_dtype=return_dtype).view(output_shape)
    if hadamard is not None and not do_transpose:
        result = rotate_hadamard_compiled(result, hadamard=hadamard)
    if bias_to_add_after is not None:
        result.add_(bias_to_add_after)
    return result


def int8_matmul_backward(
    grad_output: torch.FloatTensor,
    input: torch.FloatTensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    bias: torch.FloatTensor | None = None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    zero_point: torch.FloatTensor | None = None,
    hadamard: torch.FloatTensor | None = None,
    do_grad_input: bool = True,
    do_grad_weight: bool = True,
    do_grad_bias: bool = True,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    grad_output = grad_output.flatten(0,-2)
    if do_grad_input:
        grad_input = int8_matmul_dynamic(
            grad_output,
            dequantize_symmetric_compiled(weight, scale) if zero_point is None else dequantize_asymmetric_compiled(weight, scale, zero_point),
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            output_shape=input.shape,
            do_input_reshape=False,
        )
    if do_grad_weight:
        grad_weight = int8_matmul_dynamic(
            grad_output.t(),
            input.flatten(0,-2),
            hadamard=hadamard,
            rotate_weight=bool(hadamard is not None),
            output_shape=None,
            do_input_reshape=False,
        )
    if do_grad_bias and bias is not None:
        grad_bias = grad_output.sum(dim=0)
    return grad_input, grad_weight, grad_bias


class INT8MatmulBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor, weight: SDNQTensor, bias: torch.FloatTensor | None = None) -> torch.FloatTensor:
        ctx.save_for_backward(input, weight, bias)
        if weight.sdnq_dequantizer.use_hadamard:
            hadamard = get_hadamard(weight.sdnq_dequantizer.hadamard_group_size, dtype=input.dtype, device=input.device)
        else:
            hadamard = None

        return int8_matmul(
            input, weight.weight, weight.scale,
            bias=bias,
            svd_up=weight.svd_up,
            svd_down=weight.svd_down,
            zero_point=weight.zero_point,
            hadamard=hadamard,
            do_transpose=True,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias = ctx.saved_tensors
        if weight.sdnq_dequantizer.use_hadamard:
            hadamard = get_hadamard(weight.sdnq_dequantizer.hadamard_group_size, dtype=grad_output.dtype, device=grad_output.device)
        else:
            hadamard = None

        return int8_matmul_backward(
            grad_output, input, weight.weight, weight.scale,
            bias=bias,
            svd_up=weight.svd_up,
            svd_down=weight.svd_down,
            zero_point=weight.zero_point,
            hadamard=hadamard,
            do_grad_input=ctx.needs_input_grad[0],
            do_grad_weight=ctx.needs_input_grad[1],
            do_grad_bias=ctx.needs_input_grad[2],
        )


def quantized_linear_forward_int8_matmul(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        return quantized_linear_with_backward(input, self.weight, self.bias)
    return int8_matmul_with_backward(input, self.weight, self.bias)


int8_matmul_with_backward = INT8MatmulBackward.apply
get_int8_matmul_inputs = compile_func(get_int8_matmul_inputs)
