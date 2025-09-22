from typing import List

import torch

from .dequantizer import SDNQTensor


@torch.no_grad()
def apply_sdnq_to_module(
    model: torch.nn.Module,
    weights_dtype: str = "int8",
    group_size: int = -1,
    use_grad_ckpt: bool = True,
    use_quantized_matmul: bool = True,
    use_static_quantization: bool = True,
    use_stochastic_quantization: bool = True,
    modules_to_not_convert: List[str] = None,
):
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    quantized_forward = None

    has_children = list(model.children())
    if not has_children:
        return model

    for module_param_name, module in model.named_children():
        if module_param_name in modules_to_not_convert:
            continue

        if module.__class__.__name__ == "Linear" and hasattr(module, "weight") and module.weight is not None:
            output_channel_size, channel_size = module.weight.shape
            if channel_size >= 32 and output_channel_size >= 32:
                if weights_dtype in {"int8", "uint8"}:
                    use_quantized_matmul = use_quantized_matmul and output_channel_size % 8 == 0 and channel_size % 8 == 0
                    if use_quantized_matmul:
                        if use_grad_ckpt:
                            if use_static_quantization and group_size <= 0 and weights_dtype != "uint8":
                                from .layers.linear.linear_int8 import quantized_linear_forward_int8_matmul
                                quantized_forward = quantized_linear_forward_int8_matmul
                            else:
                                from .layers.linear.linear_int8_dynamic import quantized_linear_forward_int8_matmul_dynamic
                                quantized_forward = quantized_linear_forward_int8_matmul_dynamic
                        else:
                            if use_static_quantization and group_size <= 0 and weights_dtype != "uint8":
                                from .layers.linear.linear_int8_ckpt import quantized_linear_forward_int8_matmul_ckpt
                                quantized_forward = quantized_linear_forward_int8_matmul_ckpt
                            else:
                                from .layers.linear.linear_int8_dynamic_ckpt import quantized_linear_forward_int8_matmul_dynamic_ckpt
                                quantized_forward = quantized_linear_forward_int8_matmul_dynamic_ckpt
                    elif use_static_quantization:
                        from .layers.linear.forward import quantized_linear_forward
                        quantized_forward = quantized_linear_forward
                elif weights_dtype == "fp8":
                    use_quantized_matmul = use_quantized_matmul and output_channel_size % 16 == 0 and channel_size % 16 == 0
                    if use_quantized_matmul:
                        if use_grad_ckpt:
                            if use_static_quantization:
                                from .layers.linear.linear_fp8 import quantized_linear_forward_fp8_matmul
                                quantized_forward = quantized_linear_forward_fp8_matmul
                            else:
                                from .layers.linear.linear_fp8_dynamic import quantized_linear_forward_fp8_matmul_dynamic
                                quantized_forward = quantized_linear_forward_fp8_matmul_dynamic
                        else:
                            if use_static_quantization:
                                from .layers.linear.linear_fp8_ckpt import quantized_linear_forward_fp8_matmul_ckpt
                                quantized_forward = quantized_linear_forward_fp8_matmul_ckpt
                            else:
                                from .layers.linear.linear_fp8_dynamic_ckpt import quantized_linear_forward_fp8_matmul_dynamic_ckpt
                                quantized_forward = quantized_linear_forward_fp8_matmul_dynamic_ckpt
                    elif use_static_quantization:
                        from .layers.linear.forward import quantized_linear_forward
                        quantized_forward = quantized_linear_forward
                else:
                    raise NotImplementedError(f'Quantization type {weights_dtype} is not implemented')

                if quantized_forward is not None:
                    module.forward = quantized_forward
                    module.forward = module.forward.__get__(module, module.__class__)
                    if use_static_quantization:
                        module.weight = torch.nn.Parameter(SDNQTensor.from_float(module.weight, qtype=weights_dtype, group_size=group_size, sr=use_stochastic_quantization), requires_grad=module.weight.requires_grad)

        module = apply_sdnq_to_module(
            module,
            weights_dtype=weights_dtype,
            use_grad_ckpt=use_grad_ckpt,
            use_quantized_matmul=use_quantized_matmul,
            use_static_quantization=use_static_quantization,
            use_stochastic_quantization=use_stochastic_quantization,
            modules_to_not_convert=modules_to_not_convert,
        )
    return model
