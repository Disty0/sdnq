import copy
import torch
from torch.utils._python_dispatch import return_and_correct_aliasing
from torch._guards import detect_fake_mode

from ..utils import get_sdnq_params
from ..common import sdnq_keys
from ..quantizer import sdnq_quantize_layer_weight, sdnq_quantize_layer_weight_compiled
from ..dequantizer import SDNQDequantizer


class SDNQTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, sdnq_dequantizer: SDNQDequantizer, **parameters):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            sdnq_dequantizer.original_shape,
            strides=sdnq_dequantizer.original_stride,
            storage_offset=parameters["weight"].storage_offset(),
            dtype=sdnq_dequantizer.result_dtype,
            device=parameters["weight"].device,
        )

    def __init__(self, sdnq_dequantizer: SDNQDequantizer, **parameters):
        self.sdnq_dequantizer = sdnq_dequantizer
        for key, value in parameters.items():
            setattr(self, key, value)
        for key in sdnq_keys:
            if not hasattr(self, key):
                setattr(self, key, None)

    def dequantize(self, dtype: torch.dtype | None = None, non_svd: bool = False, non_hadamard: bool = False) -> torch.FloatTensor:
        parameters = get_sdnq_params(self)
        fake_mode = detect_fake_mode(parameters.values())
        if fake_mode is not None:
            with fake_mode:
                return self.sdnq_dequantizer(
                    **parameters,
                    skip_quantized_matmul=self.sdnq_dequantizer.use_quantized_matmul,
                    non_hadamard=non_hadamard,
                    non_svd=non_svd,
                    skip_compile=True,
                    dtype=dtype,
                )
        else:
            return self.sdnq_dequantizer(
                **parameters,
                skip_quantized_matmul=self.sdnq_dequantizer.use_quantized_matmul,
                non_hadamard=non_hadamard,
                non_svd=non_svd,
                dtype=dtype,
            )

    def __tensor_flatten__(self) -> tuple[list[str], SDNQDequantizer]:
        tensor_list = []
        metadata = self.sdnq_dequantizer
        for key in sdnq_keys:
            if getattr(self, key) is not None:
                tensor_list.append(key)
        return tensor_list, metadata

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict: dict[str, torch.Tensor], sdnq_dequantizer: SDNQDequantizer, outer_size=None, outer_stride=None) -> "SDNQTensor": # pylint: disable=unused-argument
        return SDNQTensor(sdnq_dequantizer, **tensor_data_dict)

    def __repr__(self) -> str:
        return f"SDNQTensor(sdnq_dequantizer={self.sdnq_dequantizer}, parameters={get_sdnq_params(self)})"

    @staticmethod
    def from_float(
        weight,
        layer_class_name: str | None = None,
        weights_dtype: str = "int8",
        scale_dtype: str | None = None,
        zero_point_dtype: str | None = None,
        hadamard_group_size: int = 256,
        group_size: int = 32,
        svd_rank: int = 32,
        svd_steps: int = 8,
        codebook_steps: int = 24,
        use_svd: bool = False,
        use_hadamard: bool = False,
        use_codebook: bool = False,
        use_codebook_scale: bool = False,
        use_stochastic_rounding: bool = True,
        dequantize_fp32: bool = True,
        skip_sr: bool = False,
        param_name: str | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> "SDNQTensor":
        fake_mode = detect_fake_mode(weight)
        if fake_mode is not None:
            with fake_mode:
                sdnq_dequantizer, weight_data = sdnq_quantize_layer_weight(
                    weight,
                    layer_class_name=layer_class_name,
                    weights_dtype=weights_dtype,
                    scale_dtype=scale_dtype,
                    zero_point_dtype=zero_point_dtype,
                    hadamard_group_size=hadamard_group_size,
                    group_size=group_size,
                    svd_rank=svd_rank,
                    svd_steps=svd_steps,
                    codebook_steps=codebook_steps,
                    use_svd=use_svd,
                    use_hadamard=use_hadamard,
                    use_codebook=use_codebook,
                    use_codebook_scale=use_codebook_scale,
                    use_quantized_matmul=False,
                    use_stochastic_rounding=use_stochastic_rounding,
                    dequantize_fp32=dequantize_fp32,
                    skip_sr=skip_sr,
                    torch_dtype=torch_dtype,
                    param_name=param_name,
                )
        else:
            sdnq_dequantizer, weight_data = sdnq_quantize_layer_weight_compiled(
                weight,
                layer_class_name=layer_class_name,
                weights_dtype=weights_dtype,
                scale_dtype=scale_dtype,
                zero_point_dtype=zero_point_dtype,
                hadamard_group_size=hadamard_group_size,
                group_size=group_size,
                svd_rank=svd_rank,
                svd_steps=svd_steps,
                codebook_steps=codebook_steps,
                use_svd=use_svd,
                use_hadamard=use_hadamard,
                use_codebook=use_codebook,
                use_codebook_scale=use_codebook_scale,
                use_quantized_matmul=False,
                use_stochastic_rounding=use_stochastic_rounding,
                dequantize_fp32=dequantize_fp32,
                skip_sr=skip_sr,
                torch_dtype=torch_dtype,
                param_name=param_name,
            )
        return SDNQTensor(sdnq_dequantizer, **weight_data)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs): # pylint: disable=unused-argument
        if kwargs is None:
            kwargs = {}
        if func not in op_implementations_dict:
            raise AssertionError(f"SDNQTensor does not yet support op: {func}")
        return op_implementations_dict[func](func, *args, **kwargs)

    def fsdp_pre_all_gather(self, mesh, outer_size=None, outer_stride=None, module=None, mp_policy=None) -> tuple[list[torch.Tensor], tuple[SDNQDequantizer, list[str]]]: # pylint: disable=unused-argument
        tensor_keys = []
        tensor_list = []
        for key in sdnq_keys:
            tensor = getattr(self, key, None)
            if tensor is not None:
                tensor_keys.append(key)
                tensor_list.append(tensor)
        return tensor_list, (self.sdnq_dequantizer, tensor_keys)

    def fsdp_post_all_gather(self, all_gather_outputs: tuple[torch.Tensor], metadata: tuple[SDNQDequantizer, list[str]], param_dtype: torch.dtype, *, out: torch.Tensor | None = None) -> "SDNQTensor": # pylint: disable=unused-argument
        return SDNQTensor(metadata[0], **dict(zip(metadata[1], all_gather_outputs))), all_gather_outputs


op_implementations_dict = {}
def register_op(ops: list[torch._ops.OpOverload]):
    def impl_decorator(op_impl):
        global op_implementations_dict # pylint: disable=global-variable-not-assigned # noqa: PLW0602
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
    torch.ops.aten.select.int,
])
def sdnq_generic_func(func, *args, **kwargs) -> torch.Tensor:
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
def sdnq_generic_func_(func, *args, **kwargs) -> SDNQTensor | torch.Tensor:
    input = args[0]
    args = [x.dequantize() if isinstance(x, SDNQTensor) else x for x in args]
    result = func(*args, **kwargs)
    if isinstance(input, SDNQTensor):
        input.copy_(result)
    return input


@register_op([
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.chunk.default,
    torch.ops.aten.view.default,
    torch.ops.aten.t.default,
    torch.ops.aten.as_strided.default,
])
def sdnq_generic_quantized(func, input, *args, **kwargs) -> SDNQTensor | list[SDNQTensor] | tuple[SDNQTensor]:
    sdnq_dequantizer = copy.deepcopy(input.sdnq_dequantizer)
    result = func(input.dequantize(), *args, **kwargs)
    if isinstance(result, (list, tuple)):
        return type(result)(
            SDNQTensor.from_float(
                tensor,
                layer_class_name=sdnq_dequantizer.layer_class_name if tensor.ndim != 1 else None,
                weights_dtype=sdnq_dequantizer.weights_dtype,
                scale_dtype=sdnq_dequantizer.scale_dtype,
                zero_point_dtype=sdnq_dequantizer.zero_point_dtype,
                hadamard_group_size=sdnq_dequantizer.hadamard_group_size,
                group_size=sdnq_dequantizer.group_size,
                svd_rank=sdnq_dequantizer.svd_rank,
                svd_steps=sdnq_dequantizer.svd_steps,
                codebook_steps=sdnq_dequantizer.codebook_steps,
                use_svd=input.svd_up is not None,
                use_hadamard=sdnq_dequantizer.use_hadamard,
                use_codebook=sdnq_dequantizer.use_codebook,
                use_codebook_scale=sdnq_dequantizer.use_codebook_scale,
                use_stochastic_rounding=sdnq_dequantizer.use_stochastic_rounding,
                dequantize_fp32=input.scale.dtype in {torch.float32, torch.float64},
                torch_dtype=sdnq_dequantizer.result_dtype,
                skip_sr=True,
            )
            for tensor in result
        )
    else:
        return SDNQTensor.from_float(
            result,
            layer_class_name=sdnq_dequantizer.layer_class_name if result.ndim != 1 else None,
            weights_dtype=sdnq_dequantizer.weights_dtype,
            scale_dtype=sdnq_dequantizer.scale_dtype,
            zero_point_dtype=sdnq_dequantizer.zero_point_dtype,
            hadamard_group_size=sdnq_dequantizer.hadamard_group_size,
            group_size=sdnq_dequantizer.group_size,
            svd_rank=sdnq_dequantizer.svd_rank,
            svd_steps=sdnq_dequantizer.svd_steps,
            codebook_steps=sdnq_dequantizer.codebook_steps,
            use_svd=input.svd_up is not None,
            use_hadamard=sdnq_dequantizer.use_hadamard,
            use_codebook=sdnq_dequantizer.use_codebook,
            use_codebook_scale=sdnq_dequantizer.use_codebook_scale,
            use_stochastic_rounding=sdnq_dequantizer.use_stochastic_rounding,
            dequantize_fp32=input.scale.dtype in {torch.float32, torch.float64},
            torch_dtype=sdnq_dequantizer.result_dtype,
            skip_sr=True,
        )


@register_op([torch.ops.aten.cat.default])
def sdnq_generic_multi_tensor_quantized(func, tensors: list[SDNQTensor, torch.Tensor], *args, **kwargs) -> SDNQTensor:
    use_svd = False
    dequantize_fp32 = True
    for tensor in tensors:
        if isinstance(tensor, SDNQTensor):
            sdnq_dequantizer = copy.deepcopy(tensor.sdnq_dequantizer)
            use_svd = tensor.svd_up is not None
            dequantize_fp32 = tensor.scale.dtype in {torch.float32, torch.float64}
            break

    return SDNQTensor.from_float(
        func([x.dequantize() if isinstance(x, SDNQTensor) else x for x in tensors], *args, **kwargs),
        layer_class_name=sdnq_dequantizer.layer_class_name,
        weights_dtype=sdnq_dequantizer.weights_dtype,
        scale_dtype=sdnq_dequantizer.scale_dtype,
        zero_point_dtype=sdnq_dequantizer.zero_point_dtype,
        hadamard_group_size=sdnq_dequantizer.hadamard_group_size,
        group_size=sdnq_dequantizer.group_size,
        svd_rank=sdnq_dequantizer.svd_rank,
        svd_steps=sdnq_dequantizer.svd_steps,
        codebook_steps=sdnq_dequantizer.codebook_steps,
        use_svd=use_svd,
        use_hadamard=sdnq_dequantizer.use_hadamard,
        use_codebook=sdnq_dequantizer.use_codebook,
        use_codebook_scale=sdnq_dequantizer.use_codebook_scale,
        use_stochastic_rounding=sdnq_dequantizer.use_stochastic_rounding,
        dequantize_fp32=dequantize_fp32,
        torch_dtype=sdnq_dequantizer.result_dtype,
        skip_sr=True,
    )


def apply_func_to_sdnq_params(
    sdnq_tensor: SDNQTensor,
    func: callable,
    args: tuple | None = None,
    kwargs: dict | None = None,
    sub_func: str | None = None,
    sub_func_args: tuple | None = None,
    sub_func_kwargs: dict | None = None,
    sub_func_skip_keys: set[str] | None = None,
    do_clone: bool = False,
) -> dict[str, torch.Tensor]:
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if sub_func_args is None:
        sub_func_args = ()
    if sub_func_kwargs is None:
        sub_func_kwargs = {}
    parameters = {}
    for key in sdnq_keys:
        tensor = getattr(sdnq_tensor, key, None)
        if tensor is not None:
            if do_clone:
                tensor = tensor.clone()
            tensor = func(tensor, *args, **kwargs)
            if sub_func is not None and (sub_func_skip_keys is None or key not in sub_func_skip_keys):
                tensor = getattr(tensor, sub_func)(*sub_func_args, **sub_func_kwargs)
            parameters[key] = tensor
    return parameters


@register_op([
    torch.ops.aten.detach.default,
    torch.ops.aten.clone.default,
    torch.ops.c10d_functional.all_gather_into_tensor.default,
    torch.ops._c10d_functional.all_gather_into_tensor.default,
    torch.ops.c10d_functional.wait_tensor.default,
    torch.ops._c10d_functional.wait_tensor.default,
])
def sdnq_view_ops(func, *args, **kwargs) -> SDNQTensor:
    out = SDNQTensor(copy.deepcopy(args[0].sdnq_dequantizer), **apply_func_to_sdnq_params(args[0], func, args=args[1:], kwargs=kwargs))
    return return_and_correct_aliasing(func, args, kwargs, out)


@register_op([torch.ops.aten.copy_.default])
def sdnq_copy_(func, x:  SDNQTensor | torch.Tensor, y:  SDNQTensor | torch.Tensor, *args, **kwargs) -> SDNQTensor | torch.Tensor: # pylint: disable=unused-argument
    if isinstance(x, SDNQTensor):
        if not isinstance(y, SDNQTensor):
            y = SDNQTensor.from_float(
                y,
                layer_class_name=x.sdnq_dequantizer.layer_class_name,
                weights_dtype=x.sdnq_dequantizer.weights_dtype,
                scale_dtype=x.sdnq_dequantizer.scale_dtype,
                zero_point_dtype=x.sdnq_dequantizer.zero_point_dtype,
                hadamard_group_size=x.sdnq_dequantizer.hadamard_group_size,
                group_size=x.sdnq_dequantizer.group_size,
                svd_rank=x.sdnq_dequantizer.svd_rank,
                svd_steps=x.sdnq_dequantizer.svd_steps,
                codebook_steps=x.sdnq_dequantizer.codebook_steps,
                use_svd=x.svd_up is not None,
                use_hadamard=x.sdnq_dequantizer.use_hadamard,
                use_codebook=x.sdnq_dequantizer.use_codebook,
                use_codebook_scale=x.sdnq_dequantizer.use_codebook_scale,
                use_stochastic_rounding=x.sdnq_dequantizer.use_stochastic_rounding,
                dequantize_fp32=x.scale.dtype in {torch.float32, torch.float64},
                torch_dtype=x.sdnq_dequantizer.result_dtype,
            )
        for key in sdnq_keys:
            tensor = getattr(x, key)
            if isinstance(tensor, torch.Tensor):
                tensor.copy_(getattr(y, key), *args, **kwargs)
    else:
        x.copy_(y.dequantize(), *args, **kwargs)
    return x


@register_op([torch.ops.aten._to_copy.default, torch.ops.aten.empty_like.default])
def sdnq_to_copy(func, *args, **kwargs) -> SDNQTensor:
    dtype = kwargs.pop("dtype", None)
    sdnq_dequantizer = copy.deepcopy(args[0].sdnq_dequantizer)
    if dtype is not None:
        sdnq_dequantizer.result_dtype = dtype
    out = SDNQTensor(sdnq_dequantizer, **apply_func_to_sdnq_params(args[0], func, args=args[1:], kwargs=kwargs))
    if dtype is not None:
        kwargs["dtype"] = dtype
    return return_and_correct_aliasing(func, args, kwargs, out)


@register_op([torch.ops.aten.zeros_like.default])
def sdnq_zeros_like(func, x: SDNQTensor, *args, **kwargs) -> torch.Tensor: # pylint: disable=unused-argument
    dtype = kwargs.pop("dtype", x.sdnq_dequantizer.result_dtype)
    device = kwargs.pop("device", x.device)
    return torch.zeros(x.sdnq_dequantizer.original_shape, *args, dtype=dtype, device=device, **kwargs)


@register_op([torch.ops.aten.ones_like.default])
def sdnq_ones_like(func, x: SDNQTensor, *args, **kwargs) -> torch.Tensor: # pylint: disable=unused-argument
    dtype = kwargs.pop("dtype", x.sdnq_dequantizer.result_dtype)
    device = kwargs.pop("device", x.device)
    return torch.ones(x.sdnq_dequantizer.original_shape, *args, dtype=dtype, device=device, **kwargs)


@register_op([torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar])
def sdnq_mul(func, x: SDNQTensor | torch.Tensor, y: SDNQTensor | torch.Tensor) -> SDNQTensor | torch.Tensor:
    if isinstance(x, SDNQTensor):
        sdnq_tensor, other = x, y
    else:
        sdnq_tensor, other = y, x
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if (
        func == torch.ops.aten.mul.Scalar or isinstance(other, (int,float)) or other.numel() == 1
        or (other.shape == sdnq_tensor.scale.shape and sdnq_tensor.scale_2 is None and sdnq_tensor.zero_point_scale is None)
    ):
        parameters = get_sdnq_params(sdnq_tensor)
        if parameters["scale_2"] is not None:
            parameters["scale_2"] = torch.mul(parameters["scale_2"], other)
            if parameters["scale_zero_point"] is not None:
                parameters["scale_zero_point"] = torch.mul(parameters["scale_zero_point"], other)
        else:
            parameters["scale"] = torch.mul(parameters["scale"], other)

        if parameters["zero_point_scale"] is not None:
            parameters["zero_point_scale"] = torch.mul(parameters["zero_point_scale"], other)
            if parameters["zero_point_2"] is not None:
                parameters["zero_point_2"] = torch.mul(parameters["zero_point_2"], other)
        elif parameters["zero_point"] is not None:
            parameters["zero_point"] = torch.mul(parameters["zero_point"], other)

        if parameters["svd_up"] is not None:
            parameters["svd_up"] = torch.mul(parameters["svd_up"], other)
        return sdnq_tensor.sdnq_dequantizer(**parameters, skip_quantized_matmul=sdnq_tensor.sdnq_dequantizer.use_quantized_matmul)
    else:
        return sdnq_tensor.dequantize().mul_(other)


@register_op([torch.ops.aten.mul_.Tensor, torch.ops.aten.mul_.Scalar])
def sdnq_mul_(func, x: SDNQTensor | torch.Tensor, y: SDNQTensor | torch.Tensor) -> SDNQTensor | torch.Tensor:
    if isinstance(x, SDNQTensor):
        sdnq_tensor, other, sdnq_first = x, y, True
    else:
        sdnq_tensor, other, sdnq_first = y, x, False
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if (
        sdnq_first
        and (
            func == torch.ops.aten.mul_.Scalar or isinstance(other, (int,float)) or other.numel() == 1
            or (other.shape == sdnq_tensor.scale.shape and sdnq_tensor.scale_2 is None and sdnq_tensor.zero_point_scale is None)
        )
    ):
        if sdnq_tensor.scale_2 is not None:
            sdnq_tensor.scale_2.mul_(other)
            if sdnq_tensor.scale_zero_point is not None:
                sdnq_tensor.scale_zero_point.mul_(other)
        else:
            sdnq_tensor.scale.mul_(other)

        if sdnq_tensor.zero_point_scale is not None:
            sdnq_tensor.zero_point_scale.mul_(other)
            if sdnq_tensor.zero_point_2 is not None:
                sdnq_tensor.zero_point_2.mul_(other)
        elif sdnq_tensor.zero_point is not None:
            sdnq_tensor.zero_point.mul_(other)

        if sdnq_tensor.svd_up is not None:
            sdnq_tensor.svd_up.mul_(other)
        return sdnq_tensor
    else:
        return x.copy_(sdnq_tensor.dequantize().mul_(other))


@register_op([torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar])
def sdnq_div(func, x: SDNQTensor | torch.Tensor, y: SDNQTensor | torch.Tensor) -> SDNQTensor | torch.Tensor:
    if isinstance(x, SDNQTensor):
        sdnq_tensor, other, sdnq_first = x, y, True
    else:
        sdnq_tensor, other, sdnq_first = y, x, False
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if (
        func == torch.ops.aten.div.Scalar or isinstance(other, (int,float)) or other.numel() == 1
        or (other.shape == sdnq_tensor.scale.shape and sdnq_tensor.scale_2 is None and sdnq_tensor.zero_point_scale is None)
    ):
        parameters = get_sdnq_params(sdnq_tensor)
        if parameters["scale_2"] is not None:
            parameters["scale_2"] = torch.div(parameters["scale_2"], other) if sdnq_first else torch.div(other, parameters["scale_2"])
            if parameters["scale_zero_point"] is not None:
                parameters["scale_zero_point"] = torch.div(parameters["scale_zero_point"], other) if sdnq_first else torch.div(other, parameters["scale_zero_point"])
        else:
            parameters["scale"] = torch.div(parameters["scale"], other) if sdnq_first else torch.div(other, parameters["scale"])

        if parameters["zero_point_scale"] is not None:
            parameters["zero_point_scale"] = torch.div(parameters["zero_point_scale"], other) if sdnq_first else torch.div(other, parameters["zero_point_scale"])
            if parameters["zero_point_2"] is not None:
                parameters["zero_point_2"] = torch.div(parameters["zero_point_2"], other) if sdnq_first else torch.div(other, parameters["zero_point_2"])
        elif parameters["zero_point"] is not None:
            parameters["zero_point"] = torch.div(parameters["zero_point"], other) if sdnq_first else torch.div(other, parameters["zero_point"])

        if parameters["svd_up"] is not None:
            parameters["svd_up"] = torch.div(parameters["svd_up"], other) if sdnq_first else torch.div(other, parameters["svd_up"])
        return sdnq_tensor.sdnq_dequantizer(**parameters, skip_quantized_matmul=sdnq_tensor.sdnq_dequantizer.use_quantized_matmul)
    else:
        if sdnq_first:
            return sdnq_tensor.dequantize().div_(other)
        else:
            return other.div(sdnq_tensor.dequantize())


@register_op([torch.ops.aten.div_.Tensor, torch.ops.aten.div_.Scalar])
def sdnq_div_(func, x: SDNQTensor | torch.Tensor, y: SDNQTensor | torch.Tensor) -> SDNQTensor | torch.Tensor:
    if isinstance(x, SDNQTensor):
        sdnq_tensor, other, sdnq_first = x, y, True
    else:
        sdnq_tensor, other, sdnq_first = y, x, False
    if isinstance(other, SDNQTensor):
        other = other.dequantize()
    if (
        sdnq_first
        and (
            func == torch.ops.aten.div_.Scalar or isinstance(other, (int,float)) or other.numel() == 1
            or (other.shape == sdnq_tensor.scale.shape and sdnq_tensor.scale_2 is None and sdnq_tensor.zero_point_scale is None)
        )
    ):
        if sdnq_tensor.scale_2 is not None:
            sdnq_tensor.scale_2.div_(other)
            if sdnq_tensor.scale_zero_point is not None:
                sdnq_tensor.scale_zero_point.div_(other)
        else:
            sdnq_tensor.scale.div_(other)

        if sdnq_tensor.zero_point_scale is not None:
            sdnq_tensor.zero_point_scale.div_(other)
            if sdnq_tensor.zero_point_2 is not None:
                sdnq_tensor.zero_point_2.div_(other)
        elif sdnq_tensor.zero_point is not None:
            sdnq_tensor.zero_point.div_(other)

        if sdnq_tensor.svd_up is not None:
            sdnq_tensor.svd_up.div_(other)
        return sdnq_tensor
    else:
        if sdnq_first:
            result = sdnq_tensor.dequantize().div_(other)
        else:
            result = other.div_(sdnq_tensor.dequantize())
        return x.copy_(result)


@register_op([torch.ops.c10d.send.default, torch.ops.c10d.recv_.default])
def sdnq_dist_ops(func, *args, **kwargs):
    assert len(args[0]) == 1
    return apply_func_to_sdnq_params(args[0][0], func, args=args[1:], kwargs=kwargs, sub_func="wait", sub_func_skip_keys={"weight"})["weight"]


@register_op([torch.ops.c10d.broadcast_.default])
def sdnq_dist_broadcast(func, *args, **kwargs):
    assert len(args[0]) == 1
    parameters = apply_func_to_sdnq_params(args[0][0], func, args=args[1:], kwargs=kwargs)
    weight_return = parameters["weight"][-1]
    for key, tensor in parameters.items():
        parameters[key] = tensor[0][0]
    return ([SDNQTensor(copy.deepcopy(args[0][0].sdnq_dequantizer), **parameters)], weight_return)


torch.serialization.add_safe_globals([SDNQTensor])
