from typing import List

import torch
from sdnq.common import use_tensorwise_fp8_matmul

from .dequantizer import SDNQTensor


@torch.no_grad()
def apply_sdnq_to_module(
    model: torch.nn.Module,
    weights_dtype: str = "uint8",
    quantized_matmul_dtype: str = "int8",
    group_size: int = 32,
    use_grad_ckpt: bool = True,
    use_quantized_matmul: bool = True,
    use_static_quantization: bool = True,
    use_stochastic_quantization: bool = True,
    modules_to_not_convert: List[str] = None,
):
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if group_size == 0:
        group_size = 32
    quantized_forward = None
    can_use_static_matmul = bool(use_static_quantization and group_size < 0 and weights_dtype == quantized_matmul_dtype)

    has_children = list(model.children())
    if not has_children:
        return model

    for module_param_name, module in model.named_children():
        if module_param_name in modules_to_not_convert:
            continue

        if module.__class__.__name__ == "Linear" and hasattr(module, "weight") and module.weight is not None:
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
                        module.weight = torch.nn.Parameter(SDNQTensor.from_float(module.weight, qtype=weights_dtype, group_size=group_size, sr=use_stochastic_quantization), requires_grad=module.weight.requires_grad)

        module = apply_sdnq_to_module(
            module,
            weights_dtype=weights_dtype,
            quantized_matmul_dtype=quantized_matmul_dtype,
            group_size=group_size,
            use_grad_ckpt=use_grad_ckpt,
            use_quantized_matmul=use_quantized_matmul,
            use_static_quantization=use_static_quantization,
            use_stochastic_quantization=use_stochastic_quantization,
            modules_to_not_convert=modules_to_not_convert,
        )
    return model
