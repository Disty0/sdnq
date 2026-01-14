from typing import Dict, List, Optional

import copy
import torch

from ..quantizer import SDNQConfig, QuantizationMethod, check_param_name_in, get_minimum_dtype, add_module_skip_keys
from ..dequantizer import dequantize_layer_weight
from ..loader import apply_sdnq_options_to_model
from ..common import linear_types, check_torch_compile
from ..layers.wrapper import SDNQLayer

from ..forward import get_forward_func as get_sdnq_forward_func
from .forward import get_forward_func
from .tensor import SDNQTensor


@torch.no_grad()
def apply_sdnq_training_to_module(model, weights_dtype="uint8", quantized_matmul_dtype="int8", torch_dtype=None, group_size=32, svd_rank=32, svd_steps=8, use_svd=False, use_grad_ckpt=True, use_quantized_matmul=False, use_static_quantization=True, use_stochastic_rounding=True, dequantize_fp32=True, non_blocking=False, quantization_device=None, return_device=None, modules_to_not_convert=None, modules_dtype_dict=None, full_param_name=""):
    if not use_quantized_matmul and not use_static_quantization:
        return model
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if modules_dtype_dict is None:
        modules_dtype_dict = {}

    has_children = list(model.children())
    if not has_children:
        return model

    for module_name, module in model.named_children():
        if full_param_name:
            param_name = full_param_name + "." + module_name
        else:
            param_name = module_name
        if module.__class__.__name__ == "Linear" and hasattr(module, "weight") and module.weight is not None:
            param_name = param_name + ".weight"
            if check_param_name_in(param_name, modules_to_not_convert):
                continue
            output_channel_size, channel_size = module.weight.shape

            if channel_size >= 32 and output_channel_size >= 32:
                param_weights_dtype = get_minimum_dtype(weights_dtype, param_name, modules_dtype_dict)
                if use_static_quantization:
                    if quantization_device is None:
                        quantization_device = module.weight.device
                    if return_device is None:
                        return_device = module.weight.device
                    module.weight = torch.nn.Parameter(
                        SDNQTensor.from_float(
                            module.weight.to(quantization_device, non_blocking=non_blocking),
                            layer_class_name="Linear",
                            weights_dtype=param_weights_dtype,
                            torch_dtype=torch_dtype,
                            group_size=group_size,
                            svd_rank=svd_rank,
                            svd_steps=svd_steps,
                            use_svd=use_svd,
                            use_stochastic_rounding=use_stochastic_rounding,
                            dequantize_fp32=dequantize_fp32,
                        ).to(return_device, non_blocking=non_blocking),
                        requires_grad=module.weight.requires_grad,
                    )
                    current_group_size = module.weight.sdnq_dequantizer.group_size
                else:
                    current_group_size = -1

                current_use_quantized_matmul = use_quantized_matmul and output_channel_size % 16 == 0 and channel_size % 16 == 0
                quantized_forward = get_forward_func(param_weights_dtype, quantized_matmul_dtype, use_grad_ckpt, current_use_quantized_matmul, use_static_quantization, current_group_size)

                if quantized_forward is not None:
                    module = SDNQLayer(module, quantized_forward)
                    setattr(model, module_name, module)

        setattr(model, module_name, apply_sdnq_training_to_module(
            module,
            weights_dtype=weights_dtype,
            quantized_matmul_dtype=quantized_matmul_dtype,
            group_size=group_size,
            svd_rank=svd_rank,
            use_svd=use_svd,
            use_grad_ckpt=use_grad_ckpt,
            use_quantized_matmul=use_quantized_matmul,
            use_static_quantization=use_static_quantization,
            use_stochastic_rounding=use_stochastic_rounding,
            quantization_device=quantization_device,
            return_device=return_device,
            modules_to_not_convert=modules_to_not_convert,
            full_param_name=param_name,
        ))
    return model


@torch.no_grad()
def sdnq_training_post_load_quant(
    model: torch.nn.Module,
    weights_dtype: str = "uint8",
    quantized_matmul_dtype: str = "int8",
    torch_dtype: torch.dtype = None,
    group_size: int = 32,
    svd_rank: int = 32,
    svd_steps: int = 8,
    use_svd: bool = False,
    use_grad_ckpt: bool = True,
    use_quantized_matmul: bool = False,
    use_static_quantization: bool = True,
    use_stochastic_rounding: bool = True,
    dequantize_fp32: bool = True,
    non_blocking: bool = False,
    add_skip_keys:bool = True,
    quantization_device: Optional[torch.device] = None,
    return_device: Optional[torch.device] = None,
    modules_to_not_convert: List[str] = None,
    modules_dtype_dict: Dict[str, List[str]] = None,
):
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if modules_dtype_dict is None:
        modules_dtype_dict = {}

    modules_to_not_convert = modules_to_not_convert.copy()
    modules_dtype_dict = modules_dtype_dict.copy()
    if add_skip_keys:
        model, modules_to_not_convert, modules_dtype_dict = add_module_skip_keys(model, modules_to_not_convert, modules_dtype_dict)

    quantization_config = SDNQConfig(
        weights_dtype=weights_dtype,
        quantized_matmul_dtype=quantized_matmul_dtype,
        group_size=group_size,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_svd=use_svd,
        use_grad_ckpt=use_grad_ckpt,
        quant_conv=False,
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
        modules_dtype_dict=modules_dtype_dict,
        is_training=True,
    )

    model = apply_sdnq_training_to_module(
        model,
        weights_dtype=weights_dtype,
        quantized_matmul_dtype=quantized_matmul_dtype,
        torch_dtype=torch_dtype,
        group_size=group_size,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_svd=use_svd,
        use_grad_ckpt=use_grad_ckpt,
        use_quantized_matmul=use_quantized_matmul,
        use_static_quantization=use_static_quantization,
        use_stochastic_rounding=use_stochastic_rounding,
        dequantize_fp32=dequantize_fp32,
        non_blocking=non_blocking,
        quantization_device=quantization_device,
        return_device=return_device,
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict=modules_dtype_dict,
    )

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
    sdnq_dequantizer.use_quantized_matmul = use_quantized_matmul
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
    if hasattr(model, "sdnq_dequantizer"):
        layer_class_name = model.original_class.__name__
        if layer_class_name not in linear_types:
            model = dequantize_layer_weight(model, inplace=True)
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
                model = dequantize_layer_weight(model, inplace=True)
    has_children = list(model.children())
    if not has_children:
        return model
    for module_name, module in model.named_children():
        if hasattr(module, "sdnq_dequantizer"):
            layer_class_name = module.original_class.__name__
            if layer_class_name not in linear_types:
                module = dequantize_layer_weight(module, inplace=True)
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
                    module = dequantize_layer_weight(module, inplace=True)
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
def convert_sdnq_model_to_training(model: torch.nn.Module, dtype: torch.dtype = None, quantized_matmul_dtype: str = "int8", use_grad_ckpt: bool = True, use_quantized_matmul: bool = False, use_stochastic_rounding: bool = True, dequantize_fp32: bool = True):
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
    zero_point = torch.nn.Parameter(self.weight.zero_point, requires_grad=False)
    svd_up = torch.nn.Parameter(self.weight.svd_up, requires_grad=False)
    svd_down = torch.nn.Parameter(self.weight.svd_down, requires_grad=False)
    quantized_forward = get_sdnq_forward_func(self.original_class.__name__, sdnq_dequantizer.quantized_matmul_dtype, False)
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
def convert_training_model_to_sdnq(model: torch.nn.Module, dtype: torch.dtype = None, dequantize_fp32: bool = None, use_quantized_matmul: bool = None):
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
