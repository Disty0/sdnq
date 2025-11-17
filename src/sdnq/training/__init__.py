from typing import Dict, List, Optional
from enum import Enum

import torch

from sdnq.quantizer import check_param_name_in, get_minimum_dtype, add_module_skip_keys
from sdnq.common import use_tensorwise_fp8_matmul

from .tensor import SDNQTensor


class QuantizationMethod(str, Enum):
    SDNQ_TRAINING = "sdnq_training"


@torch.no_grad()
def apply_sdnq_to_module(model, weights_dtype="uint8", quantized_matmul_dtype="int8", group_size=32, svd_rank=32, use_svd=False, use_grad_ckpt=True, use_quantized_matmul=True, use_static_quantization=True, use_stochastic_rounding=True, non_blocking=False, quantization_device=None, return_device=None, modules_to_not_convert=None, modules_dtype_dict=None, full_param_name=""):
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if modules_dtype_dict is None:
        modules_dtype_dict = {}
    if group_size == 0:
        group_size = 32
    quantized_forward = None
    can_use_static_matmul = bool(use_static_quantization and group_size < 0 and weights_dtype == quantized_matmul_dtype)

    has_children = list(model.children())
    if not has_children:
        return model

    for param_name, module in model.named_children():
        if full_param_name:
            param_name = full_param_name + "." + param_name
        if module.__class__.__name__ == "Linear" and hasattr(module, "weight") and module.weight is not None:
            param_name = param_name + ".weight"
            if check_param_name_in(param_name, modules_to_not_convert):
                continue
            output_channel_size, channel_size = module.weight.shape
            if channel_size >= 32 and output_channel_size >= 32:
                if quantized_matmul_dtype == "int8":
                    use_quantized_matmul = use_quantized_matmul and output_channel_size % 8 == 0 and channel_size % 8 == 0
                else:
                    use_quantized_matmul = use_quantized_matmul and output_channel_size % 16 == 0 and channel_size % 16 == 0
                if use_quantized_matmul:
                    if quantized_matmul_dtype == "int8":
                        if use_grad_ckpt:
                            if can_use_static_matmul:
                                from .layers.linear.linear_int8 import quantized_linear_forward_int8_matmul
                                quantized_forward = quantized_linear_forward_int8_matmul
                            else:
                                from .layers.linear.linear_int8_dynamic import quantized_linear_forward_int8_matmul_dynamic
                                quantized_forward = quantized_linear_forward_int8_matmul_dynamic
                        else:
                            if can_use_static_matmul:
                                from .layers.linear.linear_int8_ckpt import quantized_linear_forward_int8_matmul_ckpt
                                quantized_forward = quantized_linear_forward_int8_matmul_ckpt
                            else:
                                from .layers.linear.linear_int8_dynamic_ckpt import quantized_linear_forward_int8_matmul_dynamic_ckpt
                                quantized_forward = quantized_linear_forward_int8_matmul_dynamic_ckpt
                    elif quantized_matmul_dtype == "fp8":
                        if use_tensorwise_fp8_matmul:
                            if use_grad_ckpt:
                                if can_use_static_matmul:
                                    from .layers.linear.linear_fp8_tensorwise import quantized_linear_forward_fp8_matmul_tensorwise
                                    quantized_forward = quantized_linear_forward_fp8_matmul_tensorwise
                                else:
                                    from .layers.linear.linear_fp8_tensorwise_dynamic import quantized_linear_forward_fp8_matmul_tensorwise_dynamic
                                    quantized_forward = quantized_linear_forward_fp8_matmul_tensorwise_dynamic
                            else:
                                if can_use_static_matmul:
                                    from .layers.linear.linear_fp8_tensorwise_ckpt import quantized_linear_forward_fp8_matmul_tensorwise_ckpt
                                    quantized_forward = quantized_linear_forward_fp8_matmul_tensorwise_ckpt
                                else:
                                    from .layers.linear.linear_fp8_tensorwise_dynamic_ckpt import quantized_linear_forward_fp8_matmul_tensorwise_dynamic_ckpt
                                    quantized_forward = quantized_linear_forward_fp8_matmul_tensorwise_dynamic_ckpt
                        else:
                            if use_grad_ckpt:
                                if can_use_static_matmul:
                                    from .layers.linear.linear_fp8 import quantized_linear_forward_fp8_matmul
                                    quantized_forward = quantized_linear_forward_fp8_matmul
                                else:
                                    from .layers.linear.linear_fp8_dynamic import quantized_linear_forward_fp8_matmul_dynamic
                                    quantized_forward = quantized_linear_forward_fp8_matmul_dynamic
                            else:
                                if can_use_static_matmul:
                                    from .layers.linear.linear_fp8_ckpt import quantized_linear_forward_fp8_matmul_ckpt
                                    quantized_forward = quantized_linear_forward_fp8_matmul_ckpt
                                else:
                                    from .layers.linear.linear_fp8_dynamic_ckpt import quantized_linear_forward_fp8_matmul_dynamic_ckpt
                                    quantized_forward = quantized_linear_forward_fp8_matmul_dynamic_ckpt
                    else:
                        raise NotImplementedError(f'Quantized MatMul type {quantized_matmul_dtype} is not implemented')
                elif use_static_quantization:
                    from .layers.linear.forward import quantized_linear_forward
                    quantized_forward = quantized_linear_forward

                if quantized_forward is not None:
                    module.forward = quantized_forward
                    module.forward = module.forward.__get__(module, module.__class__)
                    if use_static_quantization:
                        if quantization_device is None:
                            quantization_device = module.weight.device
                        if return_device is None:
                            return_device = module.weight.device
                        module.weight = torch.nn.Parameter(
                            SDNQTensor.from_float(
                                module.weight.to(quantization_device, non_blocking=non_blocking),
                                layer_class_name="Linear",
                                weights_dtype=get_minimum_dtype(weights_dtype, param_name, modules_dtype_dict),
                                group_size=group_size,
                                svd_rank=svd_rank,
                                use_svd=use_svd,
                                use_stochastic_rounding=use_stochastic_rounding,
                            ).to(return_device, non_blocking=non_blocking),
                            requires_grad=module.weight.requires_grad,
                        )

        module = apply_sdnq_to_module(
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
        )
    return model


@torch.no_grad()
def sdnq_post_load_quant(
    model: torch.nn.Module,
    weights_dtype: str = "uint8",
    quantized_matmul_dtype: str = "int8",
    group_size: int = 32,
    svd_rank: int = 32,
    use_svd: bool = False,
    use_grad_ckpt: bool = True,
    use_quantized_matmul: bool = True,
    use_static_quantization: bool = True,
    use_stochastic_rounding: bool = True,
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
        group_size=group_size,
        svd_rank=svd_rank,
        use_svd=use_svd,
        use_grad_ckpt=use_grad_ckpt,
        use_quantized_matmul=use_quantized_matmul,
        use_static_quantization=use_static_quantization,
        use_stochastic_rounding=use_stochastic_rounding,
        non_blocking=non_blocking,
        quantization_device=quantization_device,
        return_device=return_device,
        modules_to_not_convert=modules_to_not_convert,
        modules_dtype_dict=modules_dtype_dict,
    )
    model.quantization_config = {
        "weights_dtype": weights_dtype,
        "quantized_matmul_dtype": quantized_matmul_dtype,
        "group_size": group_size,
        "svd_rank": svd_rank,
        "use_svd": use_svd,
        "use_grad_ckpt": use_grad_ckpt,
        "use_quantized_matmul": use_quantized_matmul,
        "use_static_quantization": use_static_quantization,
        "use_stochastic_rounding": use_stochastic_rounding,
        "modules_to_not_convert": modules_to_not_convert,
        "modules_dtype_dict": modules_dtype_dict,
    }

    if hasattr(model, "config"):
        try:
            model.config.quantization_config = model.quantization_config
        except Exception:
            pass
        try:
            model.config["quantization_config"] = model.quantization_config
        except Exception:
            pass
    model.quantization_method = QuantizationMethod.SDNQ_TRAINING

    return model
