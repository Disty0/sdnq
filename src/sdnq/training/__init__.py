from typing import Dict, List, Optional

import torch

from ..quantizer import SDNQConfig, QuantizationMethod, check_param_name_in, get_minimum_dtype, add_module_skip_keys

from .forward import get_forward_func
from .tensor import SDNQTensor


@torch.no_grad()
def apply_sdnq_to_module(model, weights_dtype="uint8", quantized_matmul_dtype="int8", torch_dtype=None, group_size=0, svd_rank=32, svd_steps=2, use_svd=False, use_grad_ckpt=True, use_quantized_matmul=False, use_static_quantization=True, use_stochastic_rounding=False, dequantize_fp32=True, non_blocking=False, quantization_device=None, return_device=None, modules_to_not_convert=None, modules_dtype_dict=None, full_param_name=""):
    if not use_quantized_matmul and not use_quantized_matmul:
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

                if quantized_matmul_dtype == "int8":
                    use_quantized_matmul = use_quantized_matmul and output_channel_size % 8 == 0 and channel_size % 8 == 0
                else:
                    use_quantized_matmul = use_quantized_matmul and output_channel_size % 16 == 0 and channel_size % 16 == 0
                quantized_forward = get_forward_func(param_weights_dtype, quantized_matmul_dtype, use_quantized_matmul, use_static_quantization, use_grad_ckpt, current_group_size)

                if quantized_forward is not None:
                    module.forward = quantized_forward
                    module.forward = module.forward.__get__(module, module.__class__)
                    setattr(model, module_name, module)

        setattr(model, module_name, apply_sdnq_to_module(
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
def sdnq_post_load_quant(
    model: torch.nn.Module,
    weights_dtype: str = "uint8",
    quantized_matmul_dtype: str = "int8",
    torch_dtype: torch.dtype = None,
    group_size: int = 0,
    svd_rank: int = 32,
    svd_steps: int = 2,
    use_svd: bool = False,
    use_grad_ckpt: bool = True,
    use_quantized_matmul: bool = False,
    use_static_quantization: bool = True,
    use_stochastic_rounding: bool = False,
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

    model = apply_sdnq_to_module(
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
    model.quantization_config = SDNQConfig(
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
