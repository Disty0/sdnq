from typing import Any, List, Tuple, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch._guards import detect_fake_mode

from sdnq.common import compile_func


@torch.no_grad()
def dequantize_asymmetric(weight: torch.ByteTensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None) -> torch.FloatTensor:
    result = torch.addcmul(zero_point, weight.to(dtype=scale.dtype), scale)
    if dtype is not None:
        result = result.to(dtype=dtype)
    if result_shape is not None:
        result = result.view(result_shape)
    return result


@torch.no_grad()
def dequantize_symmetric(weight: torch.CharTensor, scale: torch.FloatTensor, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None) -> torch.FloatTensor:
    result = weight.to(dtype=scale.dtype).mul_(scale)
    if dtype is not None:
        result = result.to(dtype=dtype)
    if result_shape is not None:
        result = result.view(result_shape)
    return result


@torch.no_grad()
def dequantize_symmetric_with_bias(weight: torch.CharTensor, scale: torch.FloatTensor, bias: torch.FloatTensor, dtype: Optional[torch.dtype] = None, result_shape: Optional[torch.Size] = None) -> torch.FloatTensor:
    result = torch.addcmul(bias, weight.to(dtype=scale.dtype), scale)
    if dtype is not None:
        result = result.to(dtype=dtype)
    if result_shape is not None:
        result = result.view(result_shape)
    return result


@torch.no_grad()
def quantize_int8(input: torch.FloatTensor, dim: int = -1) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    input = input.to(dtype=torch.float32)
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(127)
    input = torch.div(input, scale).round_().clamp_(-128, 127).to(dtype=torch.int8)
    return input, scale


@torch.no_grad()
def quantize_int8_sr(input: torch.FloatTensor, dim: int = -1) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    input = input.to(dtype=torch.float32)
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(127)
    input = torch.normal(0, 0.1, input.shape, device=input.device, dtype=input.dtype
    ).addcdiv_(input, scale).round_().clamp_(-128, 127).to(dtype=torch.int8)
    return input, scale


@torch.no_grad()
def quantize_uint8(input: torch.FloatTensor, dim: int = -1) -> Tuple[torch.ByteTensor, torch.FloatTensor, torch.FloatTensor]:
    input = input.to(dtype=torch.float32)
    zero_point = torch.amin(input, dim=dim, keepdims=True)
    scale = torch.amax(input, dim=dim, keepdims=True).sub_(zero_point).div_(255)
    input = torch.sub(input, zero_point).div_(scale).round_().clamp_(0, 255).to(dtype=torch.uint8)
    return input, scale, zero_point


@torch.no_grad()
def quantize_uint8_sr(input: torch.FloatTensor, dim: int = -1) -> Tuple[torch.ByteTensor, torch.FloatTensor, torch.FloatTensor]:
    input = input.to(dtype=torch.float32)
    zero_point = torch.amin(input, dim=dim, keepdims=True)
    scale = torch.amax(input, dim=dim, keepdims=True).sub_(zero_point).div_(255)
    input = torch.normal(0, 0.1, input.shape, device=input.device, dtype=input.dtype
    ).addcdiv_(torch.sub(input, zero_point), scale).round_().clamp_(0, 255).to(dtype=torch.uint8)
    return input, scale, zero_point


@torch.no_grad()
def quantize_fp8(input: torch.FloatTensor, dim: int = -1) -> Tuple[torch.Tensor, torch.FloatTensor]:
    input = input.to(dtype=torch.float32)
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(448)
    input = torch.div(input, scale).nan_to_num_().clamp_(-448, 448).to(dtype=torch.float8_e4m3fn)
    return input, scale


@torch.no_grad()
def quantize_fp8_sr(input: torch.FloatTensor, dim: int = -1) -> Tuple[torch.Tensor, torch.FloatTensor]:
    input, scale = quantize_fp8(input, dim=dim)
    input = input.view(torch.int8).add_(torch.normal(0, 0.2, input.shape, device=input.device).round_().to(torch.int8))
    input = input.clamp_(-128,126).view(torch.uint8).clamp_(0,254).view(torch.float8_e4m3fn) # clamp to (-448,448)
    return input, scale


class SDNQTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, quant_data: torch.Tensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, original_shape: torch.Size, dtype: torch.dtype, qtype: str, group_size: int, sr: bool):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            quant_data.shape,
            strides=quant_data.stride(),
            storage_offset=quant_data.storage_offset(),
            dtype=dtype,
            device=quant_data.device,
        )

    def __init__(self, quant_data: torch.Tensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, original_shape: torch.Size, dtype: torch.dtype, qtype: str, group_size: int, sr: bool):
        self.quant_data = quant_data
        self.scale = scale
        self.zero_point = zero_point
        self.original_shape = original_shape
        self.return_dtype = dtype
        self.qtype = qtype
        self.group_size = group_size
        self.sr = sr

    def dequantize(self, dtype=None):
        if dtype is None:
            dtype = self.return_dtype
        fake_mode = detect_fake_mode((self.quant_data, self.scale, self.zero_point))
        if self.zero_point is None:
            if fake_mode is not None:
                with fake_mode:
                    return dequantize_symmetric(self.quant_data, self.scale, dtype=dtype, result_shape=self.original_shape)
            return dequantize_symmetric_compiled(self.quant_data, self.scale, dtype=dtype, result_shape=self.original_shape)
        else:
            if fake_mode is not None:
                with fake_mode:
                    return dequantize_asymmetric(self.quant_data, self.scale, self.zero_point, dtype=dtype, result_shape=self.original_shape)
            return dequantize_asymmetric_compiled(self.quant_data, self.scale, self.zero_point, dtype=dtype, result_shape=self.original_shape)
    
    def __tensor_flatten__(self) -> Tuple[List[str], Any]:
        if self.zero_point is None:
            return ("quant_data", "scale"), (self.original_shape, self.return_dtype, self.qtype, self.group_size, self.sr)
        else:
            return ("quant_data", "scale", "zero_point"), (self.original_shape, self.return_dtype, self.qtype, self.group_size, self.sr)

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, extra_metadata, outer_size=None, outer_stride=None):
        original_shape, dtype, qtype, group_size, sr = extra_metadata
        return SDNQTensor(tensor_data_dict["quant_data"], tensor_data_dict["scale"], tensor_data_dict.get("zero_point", None), original_shape, dtype, qtype, group_size, sr)

    def __repr__(self):
        return f'SDNQTensor(quant_data={repr(self.quant_data)}, scale={repr(self.scale)}, zero_point={repr(self.zero_point)}, original_shape={repr(self.original_shape)}, dtype={repr(self.return_dtype)}, qtype={repr(self.qtype)}, group_size={repr(self.group_size)}, sr={repr(self.sr)})'

    @staticmethod
    def from_float(float_tensor: torch.FloatTensor, qtype: str = "int8", group_size: int = -1, sr: bool = False):
        float_tensor = float_tensor.detach()
        fake_mode = detect_fake_mode(float_tensor)
        zero_point = None
        original_shape = None
        num_of_groups = 1

        if group_size > 0:
            original_shape = float_tensor.shape
            output_channel_size, channel_size = original_shape
            if group_size >= channel_size:
                group_size = channel_size
                num_of_groups = 1
            else:
                num_of_groups = channel_size // group_size
                while num_of_groups * group_size != channel_size: # find something divisible
                    num_of_groups -= 1
                    if num_of_groups <= 1:
                        group_size = channel_size
                        num_of_groups = 1
                        break
                    group_size = channel_size // num_of_groups
            group_size = int(group_size)
            num_of_groups = int(num_of_groups)

            if num_of_groups > 1:
                # (output_channel_size, channel_size) -> (output_channel_size, num_of_groups, group_size)
                float_tensor = float_tensor.unflatten(-1, (num_of_groups, group_size))

        if qtype == "int8":
            if sr:
                if fake_mode is not None:
                    with fake_mode:
                        quant_data, scale = quantize_int8_sr(float_tensor)
                else:
                    quant_data, scale = quantize_int8_sr_compiled(float_tensor)
            else:
                if fake_mode is not None:
                    with fake_mode:
                        quant_data, scale = quantize_int8(float_tensor)
                else:
                    quant_data, scale = quantize_int8_compiled(float_tensor)
        elif qtype == "uint8":
            if sr:
                if fake_mode is not None:
                    with fake_mode:
                        quant_data, scale, zero_point = quantize_uint8_sr(float_tensor)
                else:
                    quant_data, scale, zero_point = quantize_uint8_sr_compiled(float_tensor)
            else:
                if fake_mode is not None:
                    with fake_mode:
                        quant_data, scale, zero_point = quantize_uint8(float_tensor)
                else:
                    quant_data, scale, zero_point = quantize_uint8_compiled(float_tensor)
        elif qtype == "fp8":
            if sr:
                if fake_mode is not None:
                    with fake_mode:
                        quant_data, scale = quantize_fp8_sr(float_tensor)
                else:
                    quant_data, scale = quantize_fp8_sr_compiled(float_tensor)
            else:
                if fake_mode is not None:
                    with fake_mode:
                        quant_data, scale = quantize_fp8(float_tensor)
                else:
                    quant_data, scale = quantize_fp8_compiled(float_tensor)
            if num_of_groups == 1:
                weight_stride = quant_data.stride()
                if not (weight_stride[0] == 1 and weight_stride[1] > 1):
                    quant_data = quant_data.t().contiguous().t()
        else:
            raise NotImplementedError(f'Quantization type {qtype} is not implemented')
        return SDNQTensor(quant_data, scale, zero_point, original_shape, float_tensor.dtype, qtype, group_size, sr)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        if func not in op_implementations_dict:
            raise AssertionError(f'SDNQTensor does not yet support op: {str(func)}')
        return op_implementations_dict[func](func, *args, **kwargs)

    def fsdp_pre_all_gather(self, mesh, outer_size=None, outer_stride=None, module=None, mp_policy=None):
        dtype = mp_policy.param_dtype if mp_policy is not None else self.return_dtype
        if self.zero_point is None:
            return (self.quant_data, self.scale), (self.original_shape, dtype, self.qtype, self.group_size, self.sr)
        else:
            return (self.quant_data, self.scale, self.zero_point), (self.original_shape, dtype, self.qtype, self.group_size, self.sr)

    def fsdp_post_all_gather(self, all_gather_outputs: Tuple[torch.Tensor, ...], metadata: Any, param_dtype: torch.dtype, *, out: Optional[torch.Tensor] = None):
        if len(all_gather_outputs) == 2:
            quant_data, scale = all_gather_outputs
            zero_point = None
        else:
            quant_data, scale, zero_point = all_gather_outputs
        original_shape, dtype, qtype, group_size, sr = metadata
        return SDNQTensor(quant_data, scale, zero_point, original_shape, dtype, qtype, group_size, sr), all_gather_outputs


op_implementations_dict = {}
def register_op(ops: List[torch._ops.OpOverload]):
    def impl_decorator(op_impl):
        global op_implementations_dict
        for op in ops:
            op_implementations_dict[op] = op_impl
        return op_impl
    return impl_decorator


def sdnq_generic_func_inner(func, *args, **kwargs):
    args = [x.dequantize() if isinstance(x, SDNQTensor) else x for x in args]
    return func(*args, **kwargs)
sdnq_generic_func_inner_compiled = compile_func(sdnq_generic_func_inner)

@register_op([
    torch.ops.aten.addcmul.default,
    torch.ops.aten.addcdiv.default,
    torch.ops.aten.lerp.Tensor,
    torch.ops.aten.lerp.Scalar,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.linalg_vector_norm.default,
])
def sdnq_generic_func(func, *args, **kwargs):
    fake_mode = detect_fake_mode(args)
    if fake_mode is not None:
        with fake_mode:
            return sdnq_generic_func_inner(func, *args, **kwargs)
    return sdnq_generic_func_inner_compiled(func, *args, **kwargs)


def sdnq_generic_func_inner_(func, *args, **kwargs):
    input = args[0]
    args = [x.dequantize() if isinstance(x, SDNQTensor) else x for x in args]
    result = func(*args, **kwargs)
    if isinstance(input, SDNQTensor):
        input.copy_(result)
    return input
sdnq_generic_func_inner_compiled_ = compile_func(sdnq_generic_func_inner_)

@register_op([
    torch.ops.aten.addcmul_.default,
    torch.ops.aten.addcdiv_.default,
    torch.ops.aten.lerp_.Tensor,
    torch.ops.aten.lerp_.Scalar,
    torch.ops.aten.sqrt_.default,
])
def sdnq_generic_func_(func, *args, **kwargs):
    fake_mode = detect_fake_mode(args)
    if fake_mode is not None:
        with fake_mode:
            return sdnq_generic_func_inner_(func, *args, **kwargs)
    return sdnq_generic_func_inner_compiled_(func, *args, **kwargs)


@register_op([
    torch.ops.aten.detach.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.t.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.c10d_functional.all_gather_into_tensor.default,
    torch.ops._c10d_functional.all_gather_into_tensor.default,
    torch.ops.c10d_functional.wait_tensor.default,
    torch.ops._c10d_functional.wait_tensor.default,
])
def sdnq_view_ops(func, *args, **kwargs):
    out = SDNQTensor(
        func(args[0].quant_data, *args[1:], **kwargs),
        func(args[0].scale, *args[1:], **kwargs),
        func(args[0].zero_point, *args[1:], **kwargs) if args[0].zero_point is not None else None,
        args[0].original_shape,
        args[0].return_dtype,
        args[0].qtype,
        args[0].group_size,
        args[0].sr,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@register_op([torch.ops.aten._to_copy.default])
def sdnq_to(func, *args, **kwargs):
    dtype = kwargs.pop("dtype", None)
    out = SDNQTensor(
        func(args[0].quant_data, *args[1:], **kwargs),
        func(args[0].scale, *args[1:], **kwargs),
        func(args[0].zero_point, *args[1:], **kwargs) if args[0].zero_point is not None else None,
        args[0].original_shape,
        dtype if dtype is not None else args[0].return_dtype,
        args[0].qtype,
        args[0].group_size,
        args[0].sr,
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    return return_and_correct_aliasing(func, args, kwargs, out)


@register_op([torch.ops.aten.copy_.default])
def sdnq_copy_(func, x, y, *args, **kwargs):
    if isinstance(x, SDNQTensor):
        if not isinstance(y, SDNQTensor):
            y = SDNQTensor.from_float(y, qtype=x.qtype, group_size=x.group_size, sr=x.sr)
        x.quant_data.copy_(y.quant_data, *args, **kwargs)
        x.scale.copy_(y.scale, *args, **kwargs)
        if x.zero_point is not None:
            x.zero_point.copy_(y.zero_point, *args, **kwargs)
    else:
        x.copy_(y.dequantize(), *args, **kwargs)
    return x


@register_op([torch.ops.aten.zeros_like.default])
def sdnq_zeros_like(func, x, *args, **kwargs):
    dtype = kwargs.pop("dtype", x.return_dtype)
    device = kwargs.pop("device", x.device)
    return torch.zeros(x.original_shape, *args, dtype=dtype, device=device, **kwargs)


@register_op([torch.ops.aten.ones_like.default])
def sdnq_ones_like(func, x, *args, **kwargs):
    dtype = kwargs.pop("dtype", x.return_dtype)
    device = kwargs.pop("device", x.device)
    return torch.ones(x.original_shape, *args, dtype=dtype, device=device, **kwargs)


def sdnq_mul_inner(func, x, y):
    if isinstance(x, SDNQTensor):
        input, other = x, y
    else:
        input, other = y, x
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if func == torch.ops.aten.mul.Scalar or isinstance(other, (int,float)) or other.shape == input.scale.shape or other.numel() == 1:
        if input.zero_point is None:
            return dequantize_symmetric_compiled(input.quant_data, torch.mul(input.scale, other), dtype=input.return_dtype, result_shape=input.original_shape)
        else:
            return dequantize_asymmetric_compiled(input.quant_data, torch.mul(input.scale, other), torch.mul(input.zero_point, other), dtype=input.return_dtype, result_shape=input.original_shape)
    else:
        return input.dequantize().mul_(other)
sdnq_mul_inner_compiled = compile_func(sdnq_mul_inner)

@register_op([torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar])
def sdnq_mul(func, x, y):
    fake_mode = detect_fake_mode((x,y))
    if fake_mode is not None:
        with fake_mode:
            return sdnq_mul_inner(func, x, y)
    return sdnq_mul_inner_compiled(func, x, y)


def sdnq_mul_inner_(func, x, y):
    if isinstance(x, SDNQTensor):
        input, other, sdnq_first = x, y, True
    else:
        input, other, sdnq_first = y, x, False
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if sdnq_first and (func == torch.ops.aten.mul_.Scalar or isinstance(other, (int,float)) or other.shape == input.scale.shape or other.numel() == 1):
        input.scale.mul_(other)
        if input.zero_point is not None:
            input.zero_point.mul_(other)
        return input
    else:
        return x.copy_(input.dequantize().mul_(other))
sdnq_mul_inner_compiled_ = compile_func(sdnq_mul_inner_)

@register_op([torch.ops.aten.mul_.Tensor, torch.ops.aten.mul_.Scalar])
def sdnq_mul_(func, x, y):
    fake_mode = detect_fake_mode((x,y))
    if fake_mode is not None:
        with fake_mode:
            return sdnq_mul_inner_(func, x, y)
    return sdnq_mul_inner_compiled_(func, x, y)


def sdnq_div_inner(func, x, y):
    if isinstance(x, SDNQTensor):
        input, other, sdnq_first = x, y, True
    else:
        input, other, sdnq_first = y, x, False
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if func == torch.ops.aten.div.Scalar or isinstance(other, (int,float)) or other.shape == input.scale.shape or other.numel() == 1:
        scale = torch.div(input.scale, other) if sdnq_first else torch.div(other, input.scale)
        if input.zero_point is None:
            return dequantize_symmetric_compiled(input.quant_data, scale, dtype=input.return_dtype, result_shape=input.original_shape)
        else:
            zero_point = torch.div(input.zero_point, other) if sdnq_first else torch.div(other, input.zero_point)
            return dequantize_asymmetric_compiled(input.quant_data, scale, zero_point, dtype=input.return_dtype, result_shape=input.original_shape)
    else:
        if sdnq_first:
            return input.dequantize().div_(other)
        else:
            return other.div(input.dequantize())
sdnq_div_inner_compiled = compile_func(sdnq_div_inner)

@register_op([torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar])
def sdnq_div(func, x, y):
    fake_mode = detect_fake_mode((x,y))
    if fake_mode is not None:
        with fake_mode:
            return sdnq_div_inner(func, x, y)
    return sdnq_div_inner_compiled(func, x, y)


def sdnq_div_inner_(func, x, y):
    if isinstance(x, SDNQTensor):
        input, other, sdnq_first = x, y, True
    else:
        input, other, sdnq_first = y, x, False
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if sdnq_first and (func == torch.ops.aten.div_.Scalar or isinstance(other, (int,float)) or other.shape == input.scale.shape or other.numel() == 1):
        input.scale.div_(other)
        if input.zero_point is not None:
            input.zero_point.div_(other)
        return input
    else:
        if sdnq_first:
            result = input.dequantize().div_(other)
        else:
            result = other.div_(input.dequantize())
        return x.copy_(result)
sdnq_div_inner_compiled_ = compile_func(sdnq_div_inner_)

@register_op([torch.ops.aten.div_.Tensor, torch.ops.aten.div_.Scalar])
def sdnq_div_(func, x, y):
    fake_mode = detect_fake_mode((x,y))
    if fake_mode is not None:
        with fake_mode:
            return sdnq_div_inner_(func, x, y)
    return sdnq_div_inner_compiled_(func, x, y)


def sdnq_add_inner(func, x, y):
    if isinstance(x, SDNQTensor):
        input, other = x, y
    else:
        input, other = y, x
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if input.zero_point is not None and (func == torch.ops.aten.add.Scalar or isinstance(other, (int,float)) or other.shape == input.zero_point.shape or other.numel() == 1):
        return dequantize_asymmetric_compiled(input.quant_data, input.scale, torch.add(input.zero_point, other), dtype=input.return_dtype, result_shape=input.original_shape)
    else:
        return input.dequantize().add_(other)
sdnq_add_inner_compiled = compile_func(sdnq_add_inner)

@register_op([torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar])
def sdnq_add(func, x, y):
    fake_mode = detect_fake_mode((x,y))
    if fake_mode is not None:
        with fake_mode:
            return sdnq_add_inner(func, x, y)
    return sdnq_add_inner_compiled(func, x, y)


def sdnq_add_inner_(func, x, y):
    if isinstance(x, SDNQTensor):
        input, other, sdnq_first = x, y, True
    else:
        input, other, sdnq_first = y, x, False
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if sdnq_first and input.zero_point is not None and (func == torch.ops.aten.add_.Scalar or isinstance(other, (int,float)) or other.shape == input.zero_point.shape or other.numel() == 1):
        input.zero_point.add_(other)
        return input
    else:
        return x.copy_(input.dequantize().add_(other))
sdnq_add_inner_compiled_ = compile_func(sdnq_add_inner_)

@register_op([torch.ops.aten.add_.Tensor, torch.ops.aten.add_.Scalar])
def sdnq_add_(func, x, y):
    fake_mode = detect_fake_mode((x,y))
    if fake_mode is not None:
        with fake_mode:
            return sdnq_add_inner_(func, x, y)
    return sdnq_add_inner_compiled_(func, x, y)


def sdnq_sub_inner(func, x, y):
    if isinstance(x, SDNQTensor):
        input, other, sdnq_first = x, y, True
    else:
        input, other, sdnq_first = y, x, False
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if input.zero_point is not None and (func == torch.ops.aten.sub.Scalar or isinstance(other, (int,float)) or other.shape == input.zero_point.shape or other.numel() == 1):
        zero_point = torch.sub(input.zero_point, other) if sdnq_first else torch.sub(other, input.zero_point)
        return dequantize_asymmetric_compiled(input.quant_data, input.scale, zero_point, dtype=input.return_dtype, result_shape=input.original_shape)
    else:
        if sdnq_first:
            return input.dequantize().sub_(other)
        else:
            return other.sub(input.dequantize())
sdnq_sub_inner_compiled = compile_func(sdnq_sub_inner)

@register_op([torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar])
def sdnq_sub(func, x, y):
    fake_mode = detect_fake_mode((x,y))
    if fake_mode is not None:
        with fake_mode:
            return sdnq_sub_inner(func, x, y)
    return sdnq_sub_inner_compiled(func, x, y)


def sdnq_sub_inner_(func, x, y):
    if isinstance(x, SDNQTensor):
        input, other, sdnq_first = x, y, True
    else:
        input, other, sdnq_first = y, x, False
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if sdnq_first and input.zero_point is not None and (func == torch.ops.aten.sub_.Scalar or isinstance(other, (int,float)) or other.shape == input.zero_point.shape or other.numel() == 1):
        input.zero_point.sub_(other)
        return input
    else:
        if sdnq_first:
            result = input.dequantize().sub_(other)
        else:
            result = other.sub_(input.dequantize())
        return x.copy_(result)
sdnq_sub_inner_compiled_ = compile_func(sdnq_sub_inner_)

@register_op([torch.ops.aten.sub_.Tensor, torch.ops.aten.sub_.Scalar])
def sdnq_sub_(func, x, y):
    fake_mode = detect_fake_mode((x,y))
    if fake_mode is not None:
        with fake_mode:
            return sdnq_sub_inner_(func, x, y)
    return sdnq_sub_inner_compiled_(func, x, y)


# FSDP ops
@register_op([torch.ops.aten.split.Tensor, torch.ops.aten.chunk.default])
def sdnq_split(func, weight, size, dim=0, **kwargs):
    if dim != 0:
        raise NotImplementedError("SDNQ only supports split at dim=0")
    quant_data_list = func(weight.quant_data, size, dim=dim, **kwargs)
    scale_list = func(weight.scale, size, dim=dim, **kwargs)
    if weight.zero_point is not None:
        zero_point_list = func(weight.zero_point, size, dim=dim, **kwargs)
    else:
        zero_point_list = [None for _ in range(len(quant_data_list))]
    original_shape, dtype, qtype, sr = weight.original_shape, weight.return_dtype, weight.qtype, weight.sr
    out = [SDNQTensor(quant_data, scale, zero_point, original_shape, dtype, qtype, sr) for quant_data, scale, zero_point in zip(quant_data_list, scale_list, zero_point_list)]
    return out


@register_op([torch.ops.aten.view.default, torch.ops.aten.as_strided.default])
def sdnq_view(func, *args, **kwargs):
    out = SDNQTensor(args[0].quant_data, args[0].scale, args[0].zero_point, args[0].original_shape, args[0].return_dtype, args[0].qtype, args[0].sr)
    return return_and_correct_aliasing(func, args, kwargs, out)


torch.serialization.add_safe_globals([SDNQTensor])

dequantize_asymmetric_compiled = compile_func(dequantize_asymmetric)
dequantize_symmetric_compiled = compile_func(dequantize_symmetric)
quantize_int8_sr_compiled = compile_func(quantize_int8_sr)
quantize_int8_compiled = compile_func(quantize_int8)
quantize_uint8_sr_compiled = compile_func(quantize_uint8_sr)
quantize_uint8_compiled = compile_func(quantize_uint8)
quantize_fp8_sr_compiled = compile_func(quantize_fp8_sr)
quantize_fp8_compiled = compile_func(quantize_fp8)
