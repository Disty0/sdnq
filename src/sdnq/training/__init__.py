import copy
import torch

from ..quantizer import SDNQConfig, QuantizationMethod
from ..utils import check_param_name_in, get_minimum_dtype, add_module_skip_keys
from ..loader import apply_sdnq_options_to_model
from ..common import linear_types, check_torch_compile
from ..layers import SDNQLayer, get_sdnq_wrapper_class

from ..forward import get_forward_func as get_sdnq_forward_func
from .forward import get_forward_func
from .tensor import SDNQTensor


def get_quant_kwargs(layer: torch.nn.Module, quantization_config, torch_dtype: torch.dtype | None = None, param_name: str = "", **kwargs) -> dict:
    if not isinstance(quantization_config, SDNQConfig):
        quantization_config = SDNQConfig(**quantization_config)
    layer_class_name = layer.__class__.__name__

    quant_kwargs = {
        "weights_dtype": quantization_config.weights_dtype,
        "hadamard_group_size": quantization_config.hadamard_group_size,
        "group_size": quantization_config.group_size,
        "svd_rank": quantization_config.svd_rank,
        "svd_steps": quantization_config.svd_steps,
        "use_svd": quantization_config.use_svd,
        "use_hadamard": quantization_config.use_hadamard,
        "use_stochastic_rounding": quantization_config.use_stochastic_rounding,
        "dequantize_fp32": quantization_config.dequantize_fp32,
        "use_grad_ckpt": quantization_config.use_grad_ckpt,
        "use_quantized_matmul": quantization_config.use_quantized_matmul,
        "use_static_quantization": quantization_config.use_static_quantization,
        "quantized_matmul_dtype": quantization_config.quantized_matmul_dtype,
        "non_blocking": quantization_config.non_blocking,
        "quantization_device": quantization_config.quantization_device,
        "return_device": quantization_config.return_device,
        "layer_class_name": layer_class_name,
        "torch_dtype": torch_dtype,
        "param_name": param_name,
    }

    for key, value in kwargs.items():
        quant_kwargs[key] = value

    param_key = check_param_name_in(quant_kwargs["param_name"], quantization_config.modules_quant_config.keys())
    if param_key is not None:
        for key, value in quantization_config.modules_quant_config[param_key].items():
            quant_kwargs[key] = value

    quant_kwargs["weights_dtype"] = get_minimum_dtype(quant_kwargs["weights_dtype"], quant_kwargs["param_name"], quantization_config.modules_dtype_dict)
    if check_param_name_in(quant_kwargs["param_name"], quantization_config.modules_to_not_use_matmul) is not None:
        quant_kwargs["use_quantized_matmul"] = False

    return quant_kwargs


@torch.no_grad()
def apply_sdnq_training_to_module(model, quantization_config: SDNQConfig, torch_dtype=None, full_param_name=""):
    has_children = list(model.children())
    if not has_children:
        return model, quantization_config

    for module_name, module in model.named_children():
        if full_param_name:
            param_name = full_param_name + "." + module_name
        else:
            param_name = module_name
        if hasattr(module, "weight") and module.weight is not None:
            param_name = param_name + ".weight"
            layer_class_name = module.__class__.__name__
            param_in_modules_to_not_convert = check_param_name_in(param_name, quantization_config.modules_to_not_convert)
            if (
                layer_class_name in linear_types
                and module.weight.dtype in {torch.float64, torch.float32, torch.float16, torch.bfloat16}
                and param_in_modules_to_not_convert is None
                and module.weight.shape[0] >= 32 and module.weight.shape[1] >= 32
            ):
                output_channel_size, channel_size = module.weight.shape
                quant_kwargs = get_quant_kwargs(module, quantization_config, torch_dtype=torch_dtype, param_name=param_name)
                use_grad_ckpt = quant_kwargs.pop("use_grad_ckpt")
                use_quantized_matmul = quant_kwargs.pop("use_quantized_matmul")
                use_static_quantization = quant_kwargs.pop("use_static_quantization")
                quantized_matmul_dtype = quant_kwargs.pop("quantized_matmul_dtype")
                non_blocking = quant_kwargs.pop("non_blocking")
                quantization_device = quant_kwargs.pop("quantization_device")
                return_device = quant_kwargs.pop("return_device")

                if use_static_quantization:
                    if quantization_device is None:
                        quantization_device = module.weight.device
                    if return_device is None:
                        return_device = module.weight.device
                    module.weight = torch.nn.Parameter(
                        SDNQTensor.from_float(module.weight.to(quantization_device, non_blocking=non_blocking), **quant_kwargs).to(return_device, non_blocking=non_blocking),
                        requires_grad=module.weight.requires_grad,
                    )
                    current_group_size = module.weight.sdnq_dequantizer.group_size
                else:
                    current_group_size = -1

                current_use_quantized_matmul = use_quantized_matmul and output_channel_size % 16 == 0 and channel_size % 16 == 0
                if use_quantized_matmul and not current_use_quantized_matmul:
                    quantization_config.modules_to_not_use_matmul.append(param_name)

                quantized_forward = get_forward_func(
                    quant_kwargs["weights_dtype"],
                    quantized_matmul_dtype,
                    use_grad_ckpt,
                    current_use_quantized_matmul,
                    use_static_quantization,
                    current_group_size,
                )

                if quantized_forward is not None:
                    module = get_sdnq_wrapper_class(module, quantized_forward)
                    setattr(model, module_name, module)
            elif param_in_modules_to_not_convert is None:
                quantization_config.modules_to_not_convert.append(param_name)

        module, quantization_config = apply_sdnq_training_to_module(module, quantization_config, torch_dtype=torch_dtype, full_param_name=param_name)
        setattr(model, module_name, module)
    return model, quantization_config


@torch.no_grad()
def sdnq_training_post_load_quant(
    model: torch.nn.Module,
    weights_dtype: str = "uint8",
    quantized_matmul_dtype: str = "int8",
    hadamard_group_size: int = 128,
    group_size: int = 32,
    svd_rank: int = 32,
    svd_steps: int = 8,
    use_svd: bool = False,
    use_hadamard: bool = False,
    use_grad_ckpt: bool = True,
    use_quantized_matmul: bool = False,
    use_static_quantization: bool = True,
    use_stochastic_rounding: bool = True,
    dequantize_fp32: bool = True,
    non_blocking: bool = False,
    add_skip_keys:bool = True,
    modules_to_not_convert: list[str] | None = None,
    modules_to_not_use_matmul: list[str] | None = None,
    modules_dtype_dict: dict[str, list[str]] | None = None,
    modules_quant_config: dict[str, dict] | None = None,
    quantization_device: torch.device | None = None,
    return_device: torch.device | None = None,
    torch_dtype: torch.dtype | None = None,
):
    quantization_config = SDNQConfig(
        weights_dtype=weights_dtype,
        quantized_matmul_dtype=quantized_matmul_dtype,
        hadamard_group_size=hadamard_group_size,
        group_size=group_size,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_svd=use_svd,
        use_hadamard=use_hadamard,
        use_grad_ckpt=use_grad_ckpt,
        quant_conv=False,
        quant_embedding=False,
        use_quantized_matmul=use_quantized_matmul,
        use_quantized_matmul_conv=False,
        use_static_quantization=use_static_quantization,
        use_stochastic_rounding=use_stochastic_rounding,
        dequantize_fp32=dequantize_fp32,
        non_blocking=non_blocking,
        add_skip_keys=add_skip_keys,
        quantization_device=quantization_device,
        return_device=return_device,
        modules_to_not_convert=modules_to_not_convert,
        modules_to_not_use_matmul=modules_to_not_use_matmul,
        modules_dtype_dict=modules_dtype_dict,
        modules_quant_config=modules_quant_config,
        is_training=True,
    )

    if add_skip_keys:
        model, quantization_config = add_module_skip_keys(model, quantization_config)

    model, quantization_config = apply_sdnq_training_to_module(model, quantization_config, torch_dtype=torch_dtype)

    model.quantization_config = quantization_config
    if hasattr(model, "config"):
        try:
            model.config.quantization_config = model.quantization_config
        except Exception:
            pass
        try:
            model.config["quantization_config"] = model.quantization_config.to_dict()
        except Exception:
            pass
    model.quantization_method = QuantizationMethod.SDNQ_TRAINING

    return model


@torch.no_grad()
def convert_sdnq_layer_to_training(self: torch.nn.Module, quantized_matmul_dtype: str = "int8", use_grad_ckpt: bool = True, use_quantized_matmul: bool = False, use_stochastic_rounding: bool = True, inplace: bool = False):
    assert not self.sdnq_dequantizer.use_quantized_matmul
    if inplace:
        sdnq_dequantizer = self.sdnq_dequantizer
    else:
        sdnq_dequantizer = copy.deepcopy(self.sdnq_dequantizer)
    sdnq_dequantizer.use_stochastic_rounding = use_stochastic_rounding
    weight = torch.nn.Parameter(SDNQTensor(self.weight, self.scale, self.zero_point, self.svd_up, self.svd_down, sdnq_dequantizer), requires_grad=True)
    quantized_forward = get_forward_func(sdnq_dequantizer.weights_dtype, quantized_matmul_dtype, use_grad_ckpt, use_quantized_matmul, True, sdnq_dequantizer.group_size)
    if inplace:
        self.weight = weight
        if quantized_forward is not None:
            self.forward_func = quantized_forward
        else:
            self.forward_func = getattr(torch.nn, sdnq_dequantizer.layer_class_name).forward
        del self.sdnq_dequantizer, self.scale, self.zero_point, self.svd_up, self.svd_down
        return self
    else:
        return weight, quantized_forward


@torch.no_grad()
def convert_sdnq_module_to_training(model: torch.nn.Module, quantized_matmul_dtype: str = "int8", use_grad_ckpt: bool = True, use_quantized_matmul: bool = False, use_stochastic_rounding: bool = True):
    if isinstance(model, SDNQLayer):
        layer_class_name = model.original_class.__name__
        if layer_class_name not in linear_types:
            model = model.dequantize()
        else:
            output_channel_size, channel_size = model.sdnq_dequantizer.original_shape
            if channel_size >= 32 and output_channel_size >= 32:
                current_use_quantized_matmul = use_quantized_matmul and output_channel_size % 16 == 0 and channel_size % 16 == 0
                model = convert_sdnq_layer_to_training(
                    model,
                    quantized_matmul_dtype=quantized_matmul_dtype,
                    use_grad_ckpt=use_grad_ckpt,
                    use_quantized_matmul=current_use_quantized_matmul,
                    use_stochastic_rounding=use_stochastic_rounding,
                    inplace=True,
                )
            else:
                model = model.dequantize()
    has_children = list(model.children())
    if not has_children:
        return model
    for module_name, module in model.named_children():
        if isinstance(module, SDNQLayer):
            layer_class_name = module.original_class.__name__
            if layer_class_name not in linear_types:
                module = module.dequantize()
            else:
                output_channel_size, channel_size = module.sdnq_dequantizer.original_shape
                if channel_size >= 32 and output_channel_size >= 32:
                    current_use_quantized_matmul = use_quantized_matmul and output_channel_size % 16 == 0 and channel_size % 16 == 0
                    module = convert_sdnq_layer_to_training(
                        module,
                        quantized_matmul_dtype=quantized_matmul_dtype,
                        use_grad_ckpt=use_grad_ckpt,
                        use_quantized_matmul=current_use_quantized_matmul,
                        use_stochastic_rounding=use_stochastic_rounding,
                        inplace=True,
                    )
                else:
                    module = module.dequantize()
            setattr(model, module_name, module)
        else:
            setattr(model, module_name, convert_sdnq_module_to_training(
                module,
                quantized_matmul_dtype=quantized_matmul_dtype,
                use_grad_ckpt=use_grad_ckpt,
                use_quantized_matmul=use_quantized_matmul,
                use_stochastic_rounding=use_stochastic_rounding,
            ))
    return model


@torch.no_grad()
def convert_sdnq_model_to_training(model: torch.nn.Module, dtype: torch.dtype | None = None, quantized_matmul_dtype: str = "int8", use_grad_ckpt: bool = True, use_quantized_matmul: bool = False, use_stochastic_rounding: bool = True, dequantize_fp32: bool = True):
    if use_quantized_matmul and not check_torch_compile():
        raise RuntimeError("SDNQ Quantized MatMul requires a working Triton install.")
    model = apply_sdnq_options_to_model(model, dtype=dtype, dequantize_fp32=dequantize_fp32, use_quantized_matmul=False)
    model = convert_sdnq_module_to_training(
        model,
        quantized_matmul_dtype=quantized_matmul_dtype,
        use_grad_ckpt=use_grad_ckpt,
        use_quantized_matmul=use_quantized_matmul,
        use_stochastic_rounding=use_stochastic_rounding,
    )
    model.quantization_method = QuantizationMethod.SDNQ_TRAINING
    if hasattr(model, "quantization_config"):
        model.quantization_config.quant_method = QuantizationMethod.SDNQ_TRAINING
        model.quantization_config.use_grad_ckpt = use_grad_ckpt
        model.quantization_config.use_quantized_matmul = use_quantized_matmul
        model.quantization_config.use_stochastic_rounding = use_stochastic_rounding
        model.quantization_config.dequantize_fp32 = dequantize_fp32
        model.quantization_config.is_training = True
    if hasattr(model, "config"):
        try:
            if hasattr(model.config, "quantization_config"):
                model.config.quantization_config.quant_method = QuantizationMethod.SDNQ_TRAINING
                model.config.quantization_config.use_grad_ckpt = use_grad_ckpt
                model.config.quantization_config.use_quantized_matmul = use_quantized_matmul
                model.config.quantization_config.use_stochastic_rounding = use_stochastic_rounding
                model.config.quantization_config.dequantize_fp32 = dequantize_fp32
                model.config.quantization_config.is_training = True
        except Exception:
            pass
        try:
            if hasattr(model.config, "get") and model.config.get("quantization_config", None) is not None:
                model.config["quantization_config"].quant_method = QuantizationMethod.SDNQ_TRAINING
                model.config["quantization_config"].use_grad_ckpt = use_grad_ckpt
                model.config["quantization_config"].use_quantized_matmul = use_quantized_matmul
                model.config["quantization_config"].use_stochastic_rounding = use_stochastic_rounding
                model.config["quantization_config"].dequantize_fp32 = dequantize_fp32
                model.config["quantization_config"].is_training = True
        except Exception:
            pass
    return model


@torch.no_grad()
def convert_training_layer_to_sdnq(self: torch.nn.Module, inplace: bool = False):
    if inplace:
        sdnq_dequantizer = self.weight.sdnq_dequantizer
    else:
        sdnq_dequantizer = copy.deepcopy(self.weight.sdnq_dequantizer)
    sdnq_dequantizer.use_quantized_matmul = False
    weight = torch.nn.Parameter(self.weight.weight, requires_grad=False)
    scale = torch.nn.Parameter(self.weight.scale, requires_grad=False)
    if self.weight.zero_point is not None:
        zero_point = torch.nn.Parameter(self.weight.zero_point, requires_grad=False)
    else:
        zero_point = None
    if self.weight.svd_up is not None:
        svd_up = torch.nn.Parameter(self.weight.svd_up, requires_grad=False)
        svd_down = torch.nn.Parameter(self.weight.svd_down, requires_grad=False)
    else:
        svd_up, svd_down = None, None
    quantized_forward = get_sdnq_forward_func(self.original_class.__name__, sdnq_dequantizer.quantized_matmul_dtype, sdnq_dequantizer.use_quantized_matmul)
    if inplace:
        self.weight = weight
        self.scale = scale
        self.zero_point = zero_point
        self.svd_up = svd_up
        self.svd_down = svd_down
        self.sdnq_dequantizer = sdnq_dequantizer
        self.forward_func = quantized_forward
        return self
    else:
        return weight, scale, zero_point, svd_up, svd_down, sdnq_dequantizer, quantized_forward


@torch.no_grad()
def convert_training_module_to_sdnq(model: torch.nn.Module):
    if hasattr(model, "weight") and isinstance(model.weight, SDNQTensor):
        model = convert_training_layer_to_sdnq(model, inplace=True)
    has_children = list(model.children())
    if not has_children:
        return model
    for module_name, module in model.named_children():
        if hasattr(module, "weight") and isinstance(module.weight, SDNQTensor):
            setattr(model, module_name, convert_training_layer_to_sdnq(module, inplace=True))
        else:
            setattr(model, module_name, convert_training_module_to_sdnq(module))
    return model


@torch.no_grad()
def convert_training_model_to_sdnq(model: torch.nn.Module, dtype: torch.dtype | None = None, dequantize_fp32: bool | None = None, use_quantized_matmul: bool | None = None):
    if use_quantized_matmul and not check_torch_compile():
        raise RuntimeError("SDNQ Quantized MatMul requires a working Triton install.")
    model = convert_training_module_to_sdnq(model)
    model.quantization_method = QuantizationMethod.SDNQ
    if hasattr(model, "quantization_config"):
        if use_quantized_matmul is None:
            use_quantized_matmul = model.quantization_config.use_quantized_matmul
        if dequantize_fp32 is not None:
            model.quantization_config.dequantize_fp32 = dequantize_fp32
        model.quantization_config.quant_method = QuantizationMethod.SDNQ
        model.quantization_config.is_training = False
    if hasattr(model, "config"):
        try:
            if hasattr(model.config, "quantization_config"):
                if use_quantized_matmul is None:
                    use_quantized_matmul = model.config.quantization_config.use_quantized_matmul
                if dequantize_fp32 is not None:
                    model.config.quantization_config.use_quantized_matmul = dequantize_fp32
                model.config.quantization_config.quant_method = QuantizationMethod.SDNQ
                model.config.quantization_config.is_training = False
        except Exception:
            pass
        try:
            if hasattr(model.config, "get") and model.config.get("quantization_config", None) is not None:
                if use_quantized_matmul is None:
                    use_quantized_matmul = model.config["quantization_config"].use_quantized_matmul
                if dequantize_fp32 is not None:
                    model.config["quantization_config"].dequantize_fp32 = dequantize_fp32
                model.config["quantization_config"].quant_method = QuantizationMethod.SDNQ
                model.config["quantization_config"].is_training = False
        except Exception:
            pass
    model = apply_sdnq_options_to_model(model, dtype=dtype, dequantize_fp32=dequantize_fp32, use_quantized_matmul=use_quantized_matmul)
    return model


sdnq_post_load_quant = sdnq_training_post_load_quant
