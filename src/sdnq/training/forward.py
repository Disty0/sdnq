from ..common import dtype_dict, use_tensorwise_fp8_matmul


def get_forward_func(param_weights_dtype: str, quantized_matmul_dtype: str, use_grad_ckpt: bool, use_quantized_matmul: bool, use_static_quantization: bool, current_group_size: int):
    can_use_static_matmul = bool(use_static_quantization and current_group_size < 0 and (param_weights_dtype == quantized_matmul_dtype or (param_weights_dtype in {"fp8", "float8_e4m3fn", "float8_e5m2"} and quantized_matmul_dtype in {"fp8", "float8_e4m3fn", "fp16", "float16"})))
    if use_quantized_matmul:
        if dtype_dict[quantized_matmul_dtype]["is_integer"]:
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
        else:
            if dtype_dict[quantized_matmul_dtype]["num_bits"] == 8:
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
                if use_grad_ckpt:
                    if can_use_static_matmul:
                        from .layers.linear.linear_fp16 import quantized_linear_forward_fp16_matmul
                        quantized_forward = quantized_linear_forward_fp16_matmul
                    else:
                        from .layers.linear.linear_fp16_dynamic import quantized_linear_forward_fp16_matmul_dynamic
                        quantized_forward = quantized_linear_forward_fp16_matmul_dynamic
                else:
                    if can_use_static_matmul:
                        from .layers.linear.linear_fp16_ckpt import quantized_linear_forward_fp16_matmul_ckpt
                        quantized_forward = quantized_linear_forward_fp16_matmul_ckpt
                    else:
                        from .layers.linear.linear_fp16_dynamic_ckpt import quantized_linear_forward_fp16_matmul_dynamic_ckpt
                        quantized_forward = quantized_linear_forward_fp16_matmul_dynamic_ckpt
    elif use_static_quantization:
        from .layers.linear.forward import quantized_linear_forward
        quantized_forward = quantized_linear_forward

    return quantized_forward
