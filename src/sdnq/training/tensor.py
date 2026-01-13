from typing import Any, Dict, List, Tuple, Optional

import copy
import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch._guards import detect_fake_mode

from ..quantizer import sdnq_quantize_layer_weight, sdnq_quantize_layer_weight_compiled
from ..dequantizer import SDNQDequantizer


class SDNQTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, weight: torch.Tensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, svd_up: torch.FloatTensor, svd_down: torch.FloatTensor, sdnq_dequantizer: SDNQDequantizer):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            sdnq_dequantizer.original_shape,
            strides=sdnq_dequantizer.original_stride,
            storage_offset=weight.storage_offset(),
            dtype=sdnq_dequantizer.result_dtype,
            device=weight.device,
        )

    def __init__(self, weight: torch.Tensor, scale: torch.FloatTensor, zero_point: torch.FloatTensor, svd_up: torch.FloatTensor, svd_down: torch.FloatTensor, sdnq_dequantizer: SDNQDequantizer):
        self.weight = weight
        self.sdnq_dequantizer = sdnq_dequantizer
        self.scale = scale
        self.zero_point = zero_point
        self.svd_up = svd_up
        self.svd_down = svd_down

    def dequantize(self, dtype: torch.dtype = None, non_svd: bool = False):
        if non_svd:
            svd_up, svd_down = None, None
        else:
            svd_up, svd_down = self.svd_up, self.svd_down
        fake_mode = detect_fake_mode((self.weight, self.scale, self.zero_point, svd_up, svd_down))
        if fake_mode is not None:
            with fake_mode:
                return self.sdnq_dequantizer(self.weight, self.scale, self.zero_point, svd_up, svd_down, dtype=dtype, skip_quantized_matmul=self.sdnq_dequantizer.use_quantized_matmul, skip_compile=True)
        else:
            return self.sdnq_dequantizer(self.weight, self.scale, self.zero_point, svd_up, svd_down, dtype=dtype, skip_quantized_matmul=self.sdnq_dequantizer.use_quantized_matmul)

    def __tensor_flatten__(self) -> Tuple[List[str], Any]:
        tensor_list = ["weight", "scale"]
        metadata = self.sdnq_dequantizer
        if self.zero_point is not None:
            tensor_list.append("zero_point")
        if self.svd_up is not None:
            tensor_list.append("svd_up")
            tensor_list.append("svd_down")
        return tensor_list, metadata

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict: Dict[str, torch.Tensor], sdnq_dequantizer: SDNQDequantizer, outer_size=None, outer_stride=None):
        return SDNQTensor(tensor_data_dict["weight"], tensor_data_dict["scale"], tensor_data_dict.get("zero_point", None), tensor_data_dict.get("svd_up", None), tensor_data_dict.get("svd_down", None), sdnq_dequantizer)

    def __repr__(self):
        return f"SDNQTensor(weight={repr(self.weight)}, scale={repr(self.scale)}, zero_point={repr(self.zero_point)}, svd_up={repr(self.svd_up)}, svd_down={repr(self.svd_down)}), sdnq_dequantizer={repr(self.sdnq_dequantizer)}"

    @staticmethod
    def from_float(
        weight,
        layer_class_name: str = None,
        weights_dtype: str = "int8",
        torch_dtype: torch.dtype = None,
        group_size: int = 32,
        svd_rank: int = 32,
        svd_steps: int = 8,
        use_svd: bool = False,
        use_stochastic_rounding: bool = True,
        dequantize_fp32: bool = True,
        param_name: str = None,
    ):
        fake_mode = detect_fake_mode(weight)
        if fake_mode is not None:
            with fake_mode:
                weight, scale, zero_point, svd_up, svd_down, sdnq_dequantizer = sdnq_quantize_layer_weight(
                    weight,
                    layer_class_name=layer_class_name,
                    weights_dtype=weights_dtype,
                    torch_dtype=torch_dtype,
                    group_size=group_size,
                    svd_rank=svd_rank,
                    svd_steps=svd_steps,
                    use_svd=use_svd,
                    use_quantized_matmul=False,
                    use_stochastic_rounding=use_stochastic_rounding,
                    dequantize_fp32=dequantize_fp32,
                    param_name=param_name,
                )
        else:
            weight, scale, zero_point, svd_up, svd_down, sdnq_dequantizer = sdnq_quantize_layer_weight_compiled(
                weight,
                layer_class_name=layer_class_name,
                weights_dtype=weights_dtype,
                torch_dtype=torch_dtype,
                group_size=group_size,
                svd_rank=svd_rank,
                svd_steps=svd_steps,
                use_svd=use_svd,
                use_quantized_matmul=False,
                use_stochastic_rounding=use_stochastic_rounding,
                dequantize_fp32=dequantize_fp32,
                param_name=param_name,
            )
        return SDNQTensor(weight, scale, zero_point, svd_up, svd_down, sdnq_dequantizer)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if kwargs is None:
            kwargs = {}
        if func not in op_implementations_dict:
            raise AssertionError(f"SDNQTensor does not yet support op: {str(func)}")
        return op_implementations_dict[func](func, *args, **kwargs)

    def fsdp_pre_all_gather(self, mesh, outer_size=None, outer_stride=None, module=None, mp_policy=None):
        tensor_list = [self.weight, self.scale]
        if self.zero_point is not None:
            tensor_list.append(self.zero_point)
        if self.svd_up is not None:
            tensor_list.append(self.svd_up)
            tensor_list.append(self.svd_down)
        return tensor_list, self.sdnq_dequantizer

    def fsdp_post_all_gather(self, all_gather_outputs: Tuple[torch.Tensor, ...], sdnq_dequantizer: SDNQDequantizer, param_dtype: torch.dtype, *, out: Optional[torch.Tensor] = None):
        zero_point, svd_up, svd_down = None, None, None
        if len(all_gather_outputs) == 2:
            weight, scale = all_gather_outputs
        elif len(all_gather_outputs) == 3:
            weight, scale, zero_point = all_gather_outputs
        elif len(all_gather_outputs) == 4:
            weight, scale, svd_up, svd_down = all_gather_outputs
        else:
            weight, scale, zero_point, svd_up, svd_down = all_gather_outputs
        return SDNQTensor(weight, scale, zero_point, svd_up, svd_down, sdnq_dequantizer), all_gather_outputs


op_implementations_dict = {}
def register_op(ops: List[torch._ops.OpOverload]):
    def impl_decorator(op_impl):
        global op_implementations_dict
        for op in ops:
            op_implementations_dict[op] = op_impl
        return op_impl
    return impl_decorator


@register_op([
    torch.ops.aten.eq.Tensor,
    torch.ops.aten.ne.Tensor,
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
    torch.ops.aten.slice.Tensor,
    torch.ops.c10d_functional.all_gather_into_tensor.default,
    torch.ops._c10d_functional.all_gather_into_tensor.default,
    torch.ops.c10d_functional.wait_tensor.default,
    torch.ops._c10d_functional.wait_tensor.default,
])
def sdnq_view_ops(func, *args, **kwargs):
    out = SDNQTensor(
        func(args[0].weight, *args[1:], **kwargs),
        func(args[0].scale, *args[1:], **kwargs),
        func(args[0].zero_point, *args[1:], **kwargs) if args[0].zero_point is not None else None,
        func(args[0].svd_up, *args[1:], **kwargs) if args[0].svd_up is not None else None,
        func(args[0].svd_down, *args[1:], **kwargs) if args[0].svd_down is not None else None,
        copy.deepcopy(args[0].sdnq_dequantizer),
    )
    return return_and_correct_aliasing(func, args, kwargs, out)


@register_op([torch.ops.aten.copy_.default])
def sdnq_copy_(func, x, y, *args, **kwargs):
    if isinstance(x, SDNQTensor):
        if not isinstance(y, SDNQTensor):
            y = SDNQTensor.from_float(
                y,
                layer_class_name=x.sdnq_dequantizer.layer_class_name,
                weights_dtype=x.sdnq_dequantizer.weights_dtype,
                torch_dtype=x.sdnq_dequantizer.result_dtype,
                group_size=x.sdnq_dequantizer.group_size,
                svd_rank=x.sdnq_dequantizer.svd_rank,
                svd_steps=x.sdnq_dequantizer.svd_steps,
                use_svd=x.svd_up is not None,
                use_stochastic_rounding=x.sdnq_dequantizer.use_stochastic_rounding,
                dequantize_fp32=x.scale.dtype == torch.float32,
            )
        x.weight.copy_(y.weight, *args, **kwargs)
        x.scale.copy_(y.scale, *args, **kwargs)
        if x.zero_point is not None:
            x.zero_point.copy_(y.zero_point, *args, **kwargs)
        if x.svd_up is not None:
            x.svd_up.copy_(y.svd_up, *args, **kwargs)
            x.svd_down.copy_(y.svd_down, *args, **kwargs)
    else:
        x.copy_(y.dequantize(), *args, **kwargs)
    return x


@register_op([torch.ops.aten._to_copy.default, torch.ops.aten.empty_like.default])
def sdnq_to_copy(func, *args, **kwargs):
    cast_dtype = None
    dtype = kwargs.pop("dtype", None)
    sdnq_dequantizer = copy.deepcopy(args[0].sdnq_dequantizer)
    if dtype is not None:
        sdnq_dequantizer.result_dtype = dtype
        if args[0].scale.dtype != torch.float32:
            cast_dtype = dtype
    out = SDNQTensor(
        func(args[0].weight, *args[1:], **kwargs),
        func(args[0].scale, *args[1:], dtype=cast_dtype, **kwargs),
        func(args[0].zero_point, *args[1:], dtype=cast_dtype, **kwargs) if args[0].zero_point is not None else None,
        func(args[0].svd_up, *args[1:], dtype=cast_dtype, **kwargs) if args[0].svd_up is not None else None,
        func(args[0].svd_down, *args[1:], dtype=cast_dtype, **kwargs) if args[0].svd_down is not None else None,
        sdnq_dequantizer,
    )
    if dtype is not None:
        kwargs["dtype"] = dtype
    return return_and_correct_aliasing(func, args, kwargs, out)


@register_op([torch.ops.aten.zeros_like.default])
def sdnq_zeros_like(func, x, *args, **kwargs):
    dtype = kwargs.pop("dtype", x.sdnq_dequantizer.result_dtype)
    device = kwargs.pop("device", x.device)
    return torch.zeros(x.sdnq_dequantizer.original_shape, *args, dtype=dtype, device=device, **kwargs)


@register_op([torch.ops.aten.ones_like.default])
def sdnq_ones_like(func, x, *args, **kwargs):
    dtype = kwargs.pop("dtype", x.sdnq_dequantizer.result_dtype)
    device = kwargs.pop("device", x.device)
    return torch.ones(x.sdnq_dequantizer.original_shape, *args, dtype=dtype, device=device, **kwargs)


@register_op([torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar])
def sdnq_mul(func, x, y):
    if isinstance(x, SDNQTensor):
        input, other = x, y
    else:
        input, other = y, x
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if func == torch.ops.aten.mul.Scalar or isinstance(other, (int,float)) or other.shape == input.scale.shape or other.numel() == 1:
        zero_point, svd_up, svd_down = None, None, None
        if input.zero_point is not None:
            zero_point = torch.mul(input.zero_point, other)
        if input.svd_up is not None:
            svd_up, svd_down = torch.mul(input.svd_up, other), input.svd_down
        return input.sdnq_dequantizer(input.weight, torch.mul(input.scale, other), zero_point, svd_up, svd_down, skip_quantized_matmul=input.sdnq_dequantizer.use_quantized_matmul)
    else:
        return input.dequantize().mul_(other)


@register_op([torch.ops.aten.mul_.Tensor, torch.ops.aten.mul_.Scalar])
def sdnq_mul_(func, x, y):
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
        if input.svd_up is not None:
            input.svd_up.mul_(other)
        return input
    else:
        return x.copy_(input.dequantize().mul_(other))


@register_op([torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar])
def sdnq_div(func, x, y):
    if isinstance(x, SDNQTensor):
        input, other, sdnq_first = x, y, True
    else:
        input, other, sdnq_first = y, x, False
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if func == torch.ops.aten.div.Scalar or isinstance(other, (int,float)) or other.shape == input.scale.shape or other.numel() == 1:
        scale = torch.div(input.scale, other) if sdnq_first else torch.div(other, input.scale)
        zero_point, svd_up, svd_down = None, None, None
        if input.zero_point is not None:
            zero_point = torch.div(input.zero_point, other) if sdnq_first else torch.div(other, input.zero_point)
        if input.svd_up is not None:
            svd_down = input.svd_down
            svd_up = torch.div(input.svd_up, other) if sdnq_first else torch.div(other, input.svd_up)
        return input.sdnq_dequantizer(input.weight, scale, zero_point, svd_up, svd_down, skip_quantized_matmul=input.sdnq_dequantizer.use_quantized_matmul)
    else:
        if sdnq_first:
            return input.dequantize().div_(other)
        else:
            return other.div(input.dequantize())


@register_op([torch.ops.aten.div_.Tensor, torch.ops.aten.div_.Scalar])
def sdnq_div_(func, x, y):
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
        if input.svd_up is not None:
            input.svd_up.div_(other)
        return input
    else:
        if sdnq_first:
            result = input.dequantize().div_(other)
        else:
            result = other.div_(input.dequantize())
        return x.copy_(result)


# FSDP ops
@register_op([torch.ops.aten.split.Tensor, torch.ops.aten.chunk.default])
def sdnq_split(func, input, size, dim=0, **kwargs):
    if dim != 0:
        raise NotImplementedError("SDNQ only supports split at dim=0")
    weight_list = func(input.weight, size, dim=dim, **kwargs)
    scale_list = func(input.scale, size, dim=dim, **kwargs)
    if input.zero_point is not None:
        zero_point_list = func(input.zero_point, size, dim=dim, **kwargs)
    else:
        zero_point_list = [None for _ in range(len(weight_list))]
    if input.svd_up is not None:
        svd_up_list = func(input.svd_up, size, dim=dim, **kwargs)
        svd_down_list = func(input.svd_down, size, dim=1 if dim == 0 else dim, **kwargs)
    else:
        svd_up_list = svd_down_list = [None for _ in range(len(weight_list))]
    sdnq_dequantizer = input.sdnq_dequantizer
    out = [SDNQTensor(weight, scale, zero_point, svd_up, svd_down, sdnq_dequantizer) for weight, scale, zero_point, svd_up, svd_down in zip(weight_list, scale_list, zero_point_list, svd_up_list, svd_down_list)]
    return out


@register_op([torch.ops.aten.view.default, torch.ops.aten.as_strided.default])
def sdnq_view(func, *args, **kwargs):
    out = SDNQTensor(args[0].weight, args[0].scale, args[0].zero_point, args[0].svd_up, args[0].svd_down, args[0].sdnq_dequantizer)
    return return_and_correct_aliasing(func, args, kwargs, out)


@register_op([
    torch.ops.c10d.send.default,
    torch.ops.c10d.recv_.default,
])
def sdnq_dist_ops(func, *args, **kwargs):
    assert len(args[0]) == 1
    func([args[0][0].scale], *args[1:], **kwargs).wait()
    if args[0][0].zero_point is not None:
        func([args[0][0].zero_point], *args[1:], **kwargs).wait()
    if args[0][0].svd_up is not None:
        func([args[0][0].svd_up], *args[1:], **kwargs).wait()
        func([args[0][0].svd_down], *args[1:], **kwargs).wait()
    return func([args[0][0].weight], *args[1:], **kwargs)


@register_op([torch.ops.c10d.broadcast_.default])
def sdnq_dist_broadcast(func, *args, **kwargs):
    assert len(args[0]) == 1
    weight = func([args[0][0].weight], *args[1:], **kwargs)
    return (
        [
            SDNQTensor(
                weight[0][0],
                func([args[0][0].scale], *args[1:], **kwargs)[0][0],
                func([args[0][0].zero_point], *args[1:], **kwargs)[0][0] if args[0][0].zero_point is not None else None,
                func([args[0][0].svd_up], *args[1:], **kwargs)[0][0] if args[0][0].svd_up is not None else None,
                func([args[0][0].svd_down], *args[1:], **kwargs)[0][0] if args[0][0].svd_down is not None else None,
                copy.deepcopy(args[0][0].sdnq_dequantizer),
            ),
        ],
        weight[-1],
    )


torch.serialization.add_safe_globals([SDNQTensor])
torch.serialization.add_safe_globals([SDNQDequantizer])
