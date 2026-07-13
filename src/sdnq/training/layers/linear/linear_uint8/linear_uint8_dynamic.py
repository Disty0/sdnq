import torch

from .....common import compile_func
from .....kernel_wrappers import int_scaled_mm_func, use_contiguous_int8_mm, use_contiguous_fp16_mm
from .....quant_utils import quantize_uint_mm, rotate_hadamard, get_hadamard
from ....tensor import SDNQTensor

from ..forward import check_mats, quantized_linear_with_backward


def quantize_uint_mm_matmul(
    input: torch.FloatTensor,
    weight: torch.FloatTensor,
    hadamard: torch.FloatTensor | None = None,
    do_input_reshape: bool = True,
    use_sr: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    if hadamard is not None:
        weight = rotate_hadamard(weight, hadamard=hadamard)
    if do_input_reshape:
        input = input.flatten(0,-2)
        weight = weight.t()
        if use_contiguous_int8_mm:
            weight = weight.contiguous()
    weight, scale, zero_point = quantize_uint_mm(weight.to(dtype=torch.float32), dim=0)
    input, input_scale, input_zero_point = quantize_uint_mm(input.to(dtype=torch.float32), dim=-1, use_sr=use_sr)
    return input, weight, input_scale, scale, zero_point, input_zero_point


def uint8_matmul_dynamic(
    input: torch.FloatTensor,
    weight: torch.FloatTensor,
    bias: torch.FloatTensor | None = None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    hadamard: torch.FloatTensor | None = None,
    output_shape: torch.Size = None,
    do_input_reshape: bool = True,
    rotate_weight: bool = False,
    use_sr: bool = False,
) -> torch.FloatTensor:
    return_dtype = input.dtype
    bias_to_add_after = None
    if output_shape is None:
        output_shape = list(input.shape)
        output_shape[-1] = weight.shape[0] if do_input_reshape else weight.shape[-1]

    if hadamard is not None:
        if do_input_reshape:
            input = rotate_hadamard(input, hadamard=hadamard)
        else:
            bias_to_add_after = bias
            bias = None
    if svd_up is not None:
        input = input.flatten(0,-2)
        svd_up, svd_down = svd_up.to(dtype=return_dtype), svd_down.to(dtype=return_dtype)
        if do_input_reshape:
            if use_contiguous_fp16_mm:
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

    input, weight, input_scale, scale, zero_point, input_zero_point = quantize_uint_mm_matmul(
        input, weight,
        do_input_reshape=do_input_reshape,
        hadamard=hadamard if rotate_weight else None,
        use_sr=use_sr,
    )
    zero_bias = torch.sum(input, dim=-1, keepdim=True, dtype=torch.int32).to(input_scale.dtype).mul_(input_scale).mul(zero_point)
    zero_bias.add_(torch.sum(weight, dim=0, keepdim=True, dtype=torch.int32).to(scale.dtype).mul_(scale).mul(input_zero_point))
    zero_bias.add_(torch.mul(input_zero_point.mul_(input.shape[-1]), zero_point))
    if bias is not None:
        zero_bias.add_(bias)
    input, weight = check_mats(input, weight)

    result = int_scaled_mm_func(input, weight, input_scale, scale, bias=zero_bias, out_dtype=return_dtype).view(output_shape)
    if hadamard is not None and not do_input_reshape:
        result = rotate_hadamard(result, hadamard=hadamard)
    if bias_to_add_after is not None:
        result.add_(bias_to_add_after)
    return result


def uint8_matmul_dynamic_backward(
    grad_output: torch.FloatTensor,
    input: torch.FloatTensor,
    weight: torch.FloatTensor,
    bias: torch.FloatTensor | None = None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    hadamard: torch.FloatTensor | None = None,
    do_grad_input: bool = True,
    do_grad_weight: bool = True,
    do_grad_bias: bool = True,
) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    grad_input = grad_weight = grad_bias = None
    grad_output = grad_output.flatten(0,-2)
    if do_grad_input:
        grad_input = uint8_matmul_dynamic(
            grad_output,
            weight,
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            output_shape=input.shape,
            do_input_reshape=False,
        )
    if do_grad_weight:
        grad_weight = uint8_matmul_dynamic(
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


class UINT8MatmulDynamicBackward(torch.autograd.Function):
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

        ctx.save_for_backward(input, weight, bias, svd_up, svd_down)
        return uint8_matmul_dynamic_compiled(
            input, weight,
            bias=bias,
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            )

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        input, weight, bias, svd_up, svd_down = ctx.saved_tensors
        if ctx.use_hadamard:
            hadamard = get_hadamard(ctx.hadamard_group_size, dtype=grad_output.dtype, device=grad_output.device)
        else:
            hadamard = None

        return uint8_matmul_dynamic_backward(
            grad_output, input, weight,
            bias=bias,
            svd_up=svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            do_grad_input=ctx.needs_input_grad[0],
            do_grad_weight=ctx.needs_input_grad[1],
            do_grad_bias=ctx.needs_input_grad[2],
        )


def quantized_linear_forward_uint8_matmul_dynamic(self, input: torch.FloatTensor) -> torch.FloatTensor:
    if torch.numel(input) / input.shape[-1] < 32:
        if isinstance(self.weight, SDNQTensor):
            return quantized_linear_with_backward(input, self.weight, self.bias)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)
    return uint8_matmul_dynamic_with_backward(input, self.weight, self.bias)


uint8_matmul_dynamic_with_backward = UINT8MatmulDynamicBackward.apply
uint8_matmul_dynamic_compiled = compile_func(uint8_matmul_dynamic)
uint8_matmul_dynamic_backward = compile_func(uint8_matmul_dynamic_backward)
