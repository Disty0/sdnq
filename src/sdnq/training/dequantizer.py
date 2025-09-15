from typing import Any, List, Tuple, Optional

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from sdnq.common import use_torch_compile


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
    input = input.add(1e-8) if input.dtype == torch.float32 else input.to(dtype=torch.float32).add_(1e-8)
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(127)
    input = torch.div(input, scale).round_().clamp_(-128, 127).to(dtype=torch.int8)
    return input, scale


@torch.no_grad()
def quantize_int8_sr(input: torch.FloatTensor, dim: int = -1) -> Tuple[torch.CharTensor, torch.FloatTensor]:
    input = input.add(1e-8) if input.dtype == torch.float32 else input.to(dtype=torch.float32).add_(1e-8)
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(127)
    input = torch.normal(0, 0.1, input.shape, device=input.device, dtype=input.dtype
    ).addcdiv_(input, scale).round_().clamp_(-128, 127).to(dtype=torch.int8)
    return input, scale


@torch.no_grad()
def quantize_fp8(input: torch.FloatTensor, dim: int = -1) -> Tuple[torch.Tensor, torch.FloatTensor]:
    input = input.add(1e-8) if input.dtype == torch.float32 else input.to(dtype=torch.float32).add_(1e-8)
    scale = torch.amax(input.abs(), dim=dim, keepdims=True).div_(448)
    input = torch.div(input, scale).clamp_(-448, 448).to(dtype=torch.float8_e4m3fn)
    return input, scale


@torch.no_grad()
def quantize_fp8_sr(input: torch.FloatTensor, dim: int = -1) -> Tuple[torch.Tensor, torch.FloatTensor]:
    input, scale = quantize_fp8(input, dim=dim)
    input = input.view(torch.int8).add_(torch.normal(0, 0.2, input.shape, device=input.device).round_().to(torch.int8))
    input = input.clamp_(-128,126).view(torch.uint8).clamp_(0,254).view(torch.float8_e4m3fn) # clamp to (-448,448)
    return input, scale


class SDNQTensor(torch.Tensor):
    @staticmethod
    @torch._dynamo.disable
    def __new__(cls, quant_data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, qtype: str = "int8", sr: bool = False):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            quant_data.shape,
            strides=quant_data.stride(),
            storage_offset=quant_data.storage_offset(),
            dtype=dtype,
            device=quant_data.device,
        )

    @torch._dynamo.disable
    def __init__(self, quant_data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, qtype: str = "int8", sr: bool = False):
        self.quant_data = quant_data
        self.scale = scale
        self.return_dtype = dtype
        self.qtype = qtype
        self.sr = sr

    def dequantize(self, dtype=None):
        if dtype is None:
            dtype = self.return_dtype
        return dequantize_symmetric_compiled(self.quant_data, self.scale, dtype=dtype)
    
    def __tensor_flatten__(self) -> Tuple[List[str], Any]:
        return ["quant_data", "scale", "return_dtype", "qtype", "sr"], None

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, extra_metadata, outer_size=None, outer_stride=None):
        assert extra_metadata is None
        return SDNQTensor(
            tensor_data_dict["quant_data"],
            tensor_data_dict["scale"],
            tensor_data_dict["return_dtype"],
            tensor_data_dict["qtype"],
            sr=tensor_data_dict["sr"],
        )

    def __repr__(self):
        return f'SDNQTensor(quant_data={repr(self.quant_data)}, scale={repr(self.scale)}, dtype={self.return_dtype}, qtype={self.qtype}, sr={self.sr})'

    @staticmethod
    def from_float(float_tensor: torch.FloatTensor, qtype: str = "int8", sr: bool = False):
        if qtype == "int8":
            if sr:
                quant_data, scale = quantize_int8_sr_compiled(float_tensor.detach())
            else:
                quant_data, scale = quantize_int8_compiled(float_tensor.detach())
        elif qtype == "fp8":
            if sr:
                quant_data, scale = quantize_fp8_sr_compiled(float_tensor.detach())
            else:
                quant_data, scale = quantize_fp8_compiled(float_tensor.detach())
            weight_stride = quant_data.stride()
            if not (weight_stride[0] == 1 and weight_stride[1] > 1):
                quant_data = quant_data.t().contiguous().t()
        else:
            raise NotImplementedError(f'Quantization type {qtype} is not implemented')
        return SDNQTensor(quant_data, scale, float_tensor.dtype, qtype=qtype, sr=sr)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        if func not in op_implementations_dict:
            raise AssertionError(f'SDNQTensor does not yet support op: {str(func)}')
        return op_implementations_dict[func](func, *args, **kwargs)

    def fsdp_pre_all_gather(self, mesh, outer_size=None, outer_stride=None, module=None, mp_policy=None):
        dtype = mp_policy.param_dtype if mp_policy is not None else self.return_dtype
        return (self.quant_data, self.scale, dtype, self.qtype, self.sr), None

    def fsdp_post_all_gather(self, all_gather_outputs: Tuple[torch.Tensor, ...], metadata: Any, param_dtype: torch.dtype, *, out: Optional[torch.Tensor] = None):
        quant_data, scale, dtype, qtype, sr = all_gather_outputs
        return SDNQTensor(quant_data, scale, dtype, qtype=qtype, sr=sr), all_gather_outputs


op_implementations_dict = {}
def register_op(ops: List[torch._ops.OpOverload]):
    def impl_decorator(op_impl):
        global op_implementations_dict
        for op in ops:
            op_implementations_dict[op] = op_impl
        return op_impl
    return impl_decorator


@register_op([
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sub.Scalar,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.addcmul.default,
    torch.ops.aten.addcdiv.default,
    torch.ops.aten.lerp.Tensor,
    torch.ops.aten.lerp.Scalar,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.linalg_vector_norm.default,
])
def sdnq_generic_func(func, *args, **kwargs):
    args = [x.dequantize() if isinstance(x, SDNQTensor) else x for x in args]
    return func(*args, **kwargs)


@register_op([
    torch.ops.aten.sub_.Tensor,
    torch.ops.aten.sub_.Scalar,
    torch.ops.aten.add_.Tensor,
    torch.ops.aten.add_.Scalar,
    torch.ops.aten.addcmul_.default,
    torch.ops.aten.addcdiv_.default,
    torch.ops.aten.lerp_.Tensor,
    torch.ops.aten.lerp_.Scalar,
    torch.ops.aten.sqrt_.default,
])
def sdnq_generic_func_(func, *args, **kwargs):
    input = args[0]
    args = [x.dequantize() if isinstance(x, SDNQTensor) else x for x in args]
    result = func(*args, **kwargs)
    if isinstance(input, SDNQTensor):
        input.copy_(result)
    return input


@register_op([
    torch.ops.aten.detach.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.t.default,
    # FSDP ops
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
        args[0].return_dtype,
        qtype=args[0].qtype,
        sr=args[0].sr,
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@register_op([torch.ops.aten._to_copy.default])
def sdnq_to(func, *args, **kwargs):
    dtype = kwargs.pop("dtype", None)
    out = SDNQTensor(
        func(args[0].quant_data, *args[1:], **kwargs),
        func(args[0].scale, *args[1:], **kwargs),
        dtype if dtype is not None else args[0].return_dtype,
        qtype=args[0].qtype,
        sr=args[0].sr,
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    return return_and_correct_aliasing(func, args, kwargs, out)


@register_op([torch.ops.aten.copy_.default])
def sdnq_copy_(func, x, y, *args, **kwargs):
    if isinstance(x, SDNQTensor):
        if not isinstance(y, SDNQTensor):
            y = SDNQTensor.from_float(y, qtype=x.qtype, sr=x.sr)
        x.quant_data.copy_(y.quant_data, *args, **kwargs)
        x.scale.copy_(y.scale, *args, **kwargs)
    else:
        x.copy_(y.dequantize(), *args, **kwargs)
    return x


@register_op([torch.ops.aten.zeros_like.default])
def sdnq_zeros_like(func, x, *args, **kwargs):
    dtype = kwargs.pop("dtype", x.return_dtype)
    device = kwargs.pop("device", x.device)
    return torch.zeros(x.shape, *args, dtype=dtype, device=device, **kwargs)


@register_op([torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar])
def sdnq_mul(func, x, y):
    if isinstance(x, SDNQTensor):
        input = x
        other = y
    else:
        input = y
        other = x
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if func == torch.ops.aten.mul.Scalar or isinstance(other, (int,float)) or other.shape == input.scale.shape or other.numel() == 1:
        return dequantize_symmetric_compiled(input.quant_data, torch.mul(input.scale, other), dtype=input.return_dtype)
    else:
        return input.dequantize().mul_(other)


@register_op([torch.ops.aten.mul_.Tensor, torch.ops.aten.mul_.Scalar])
def sdnq_mul_(func, x, y):
    if isinstance(x, SDNQTensor):
        sdnq_first = True
        input = x
        other = y
    else:
        sdnq_first = False
        input = y
        other = x
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if sdnq_first and (func == torch.ops.aten.mul_.Scalar or isinstance(other, (int,float)) or other.shape == input.scale.shape or other.numel() == 1):
        input.scale.mul_(other)
        return input
    else:
        result = input.dequantize().mul_(other)
        return x.copy_(result)


@register_op([torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar])
def sdnq_div(func, x, y):
    if isinstance(x, SDNQTensor):
        sdnq_first = True
        input = x
        other = y
    else:
        sdnq_first = False
        input = y
        other = x
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if func == torch.ops.aten.div.Scalar or isinstance(other, (int,float)) or other.shape == input.scale.shape or other.numel() == 1:
        scale = torch.div(input.scale, other) if sdnq_first else torch.div(other, input.scale)
        return dequantize_symmetric_compiled(input.quant_data, scale, dtype=input.return_dtype)
    else:
        return input.dequantize().div_(other)


@register_op([torch.ops.aten.div_.Tensor, torch.ops.aten.div_.Scalar])
def sdnq_div_(func, x, y):
    if isinstance(x, SDNQTensor):
        sdnq_first = True
        input = x
        other = y
    else:
        sdnq_first = False
        input = y
        other = x
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if sdnq_first and (func == torch.ops.aten.div_.Scalar or isinstance(other, (int,float)) or other.shape == input.scale.shape or other.numel() == 1):
        input.scale.div_(other)
        return input
    else:
        result = input.dequantize().div_(other)
        return x.copy_(result)


# FSDP ops
@register_op([torch.ops.aten.split.Tensor, torch.ops.aten.chunk.default])
def sdnq_split(func, weight, size, dim=0, **kwargs):
    if dim != 0:
        raise NotImplementedError("SDNQ only supports split at dim=0")
    quant_data_list = func(weight.quant_data, size, dim=dim, **kwargs)
    scale_list = func(weight.scale, size, dim=dim, **kwargs)
    dtype, qtype, sr = weight.return_dtype, weight.qtype, weight.sr
    out = [SDNQTensor(quant_data, scale, dtype, qtype=qtype, sr=sr) for quant_data, scale in zip(quant_data_list, scale_list)]
    return out


@register_op([torch.ops.aten.new_zeros.default])
def sdnq_new_zeros(func, x, size, *args, **kwargs):
    device = kwargs.pop("device", x.device)
    dtype = kwargs.pop("dtype", x.return_dtype)
    quant_data = torch.zeros(size, device=device, dtype=torch.int8)
    scale = torch.zeros((*size[:-1],1), device=device, dtype=torch.float32)
    return SDNQTensor(quant_data, scale, dtype, qtype=x.qtype, sr=x.sr)


@register_op([torch.ops.aten.view.default, torch.ops.aten.as_strided.default])
def sdnq_view(func, *args, **kwargs):
    out = SDNQTensor(args[0].quant_data, args[0].scale, args[0].return_dtype, qtype=args[0].qtype, sr=args[0].sr)
    return return_and_correct_aliasing(func, args, kwargs, out)


torch.serialization.add_safe_globals([SDNQTensor])

if use_torch_compile:
    dequantize_symmetric_compiled = torch.compile(dequantize_symmetric, fullgraph=True, dynamic=False)
    quantize_int8_sr_compiled = torch.compile(quantize_int8_sr, fullgraph=True, dynamic=False)
    quantize_int8_compiled = torch.compile(quantize_int8, fullgraph=True, dynamic=False)
    quantize_fp8_sr_compiled = torch.compile(quantize_fp8_sr, fullgraph=True, dynamic=False)
    quantize_fp8_compiled = torch.compile(quantize_fp8, fullgraph=True, dynamic=False)
else:
    dequantize_symmetric_compiled = dequantize_symmetric
    quantize_int8_sr_compiled = quantize_int8_sr
    quantize_int8_compiled = quantize_int8
    quantize_fp8_sr_compiled = quantize_fp8_sr
    quantize_fp8_compiled = quantize_fp8
