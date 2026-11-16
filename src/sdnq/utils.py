import re
import torch

from .common import (
    sdnq_keys,
    dtype_dict,
    torch_device,
    common_skip_keys,
    module_skip_keys_dict,
    allowed_types,
    embedding_types,
    conv_types,
    conv_transpose_types,
)


def is_pow2(n: int) -> bool:
    assert n > 0 and isinstance(n, int), "n must be a positive integer"
    return (n & (n - 1)) == 0


def is_pow4(n: int) -> bool:
    return is_pow2(n) and (n.bit_length() & 1 == 1)


def next_power_of_2(n: int) -> int:
    if is_pow2(n):
        return n
    return 2 ** n.bit_length()


def prev_power_of_2(n: int) -> int:
    if is_pow2(n):
        return n
    return 1 << (n.bit_length() - 1)


def normalize_device(device: torch.device) -> torch.device:
    if device is None:
        return None
    if torch.device(device).type in {"cpu", "mps", "meta"}:
        return torch.device(device)
    if torch.device(device).index is None:
        return torch.device(str(device), index=0)
    return torch.device(device)


def check_same_device(d1: torch.device, d2: torch.device) -> bool:
    return normalize_device(d1) == normalize_device(d2)


def check_tma_alignment(tensor: torch.Tensor, alignment: int = 16, dim: int | None = None) -> bool:
    if tensor is None:
        return True
    if dim is None:
        dim = -1
        while tensor.shape[dim] == 1 and dim > -tensor.ndim:
            dim -= 1
    return bool((tensor.shape[dim] % (alignment // tensor.element_size())) == 0)


def get_smem_size(dev: int) -> int:
    try:
        import triton
        return triton.runtime.driver.active.utils.get_device_properties(dev).get("max_shared_mem", 0)
    except Exception:
        return 0


def get_cache_sizes(device: torch.device) -> tuple[int, int]:
    device = torch.device(device)
    device_idx = device.index if device.index is not None else 0
    match device.type:
        case "cuda":
            cache_size = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0)
            smem_size = get_smem_size(device_idx)
        case "xpu":
            props = torch.xpu.get_device_properties(device)
            cache_size = getattr(props, "last_level_cache_size", 0)
            smem_size = getattr(props, "local_mem_size", 0)
            if smem_size == 0:
                smem_size = get_smem_size(device_idx)
        case _:
            cache_size = 0
            smem_size = get_smem_size(device_idx)
    return cache_size, smem_size


def get_warp_props(device: torch.device, occupancy_factor: int = 4) -> dict[str, int]:
    device = torch.device(device)
    device_idx = device.index if device.index is not None else 0

    try:
        import triton
        triton_props = triton.runtime.driver.active.utils.get_device_properties(device_idx) or {}
    except Exception:
        triton_props = {}
    try:
        if device.type == "xpu":
            torch_props = torch.xpu.get_device_properties(device_idx)
        elif device.type == "cuda":
            torch_props = torch.cuda.get_device_properties(device_idx)
        else:
            torch_props = None
    except Exception:
        torch_props = None

    max_work_group_size = (
        triton_props.get("max_work_group_size")
        or getattr(torch_props, "max_work_group_size", None)
        or getattr(torch_props, "max_threads_per_block", 0)
        or 1024
    )
    max_threads_per_sm = (
        triton_props.get("max_threads_per_sm")
        or getattr(torch_props, "max_threads_per_multi_processor", 0)
        or 1024
    )
    max_regs_per_sm = (
        triton_props.get("max_num_regs")
        or getattr(torch_props, "regs_per_multiprocessor", 0)
        or (131072 if device.type == "xpu" else 65536)
    )

    if device.type == "xpu":
        max_work_group_size = max_work_group_size // 2 # 256 GRF mode
        warp_size = min(triton_props.get("sub_group_sizes") or getattr(torch_props, "sub_group_sizes", None) or (16, 32))
    else:
        warp_size = triton_props.get("warpSize") or getattr(torch_props, "warp_size", 0) or 32

    max_warps = max_work_group_size // warp_size

    if device.type == "xpu":
        max_warps = min(max_warps, getattr(torch_props, "max_num_sub_groups", float("inf")))
    else:
        max_warps = min(max_warps, max_threads_per_sm // warp_size)

    max_warps = prev_power_of_2(max(1, max_warps))
    min_warps = min(2, max_warps)
    target_max_warps = min(max_warps, max(4, prev_power_of_2(max(1, max_threads_per_sm // warp_size // occupancy_factor))))
    register_threshold = 256 * warp_size

    return {
        "warp_size": warp_size,
        "min_warps": min_warps,
        "max_warps": max_warps,
        "target_max_warps": target_max_warps,
        "register_threshold": register_threshold,
        "max_regs_per_sm": max_regs_per_sm,
    }


def get_num_warps(
    block_size: int,
    register_req: int = 0,
    warp_props: dict[str, int] | None = None,
    device: torch.device | None = None,
    return_prune: bool = False,
) -> int | tuple[int, bool]:
    if warp_props is None:
        warp_props = get_warp_props(device if device is not None else torch_device)
    num_warps = max(
        warp_props["min_warps"],
        min(
            warp_props["target_max_warps"],
            next_power_of_2((max(block_size, warp_props["warp_size"]) + warp_props["warp_size"] - 1) // warp_props["warp_size"]),
        ),
    )
    while num_warps < warp_props["max_warps"] and (register_req / num_warps) > warp_props["register_threshold"]:
        num_warps *= 2
    num_warps = max(warp_props["min_warps"], min(num_warps, warp_props["max_warps"]))
    if return_prune:
        do_prune = bool(register_req > warp_props["max_regs_per_sm"] or (register_req / num_warps) > warp_props["register_threshold"])
        return num_warps, do_prune
    return num_warps


def get_module_from_name(module: torch.nn.Module, tensor_name: str) -> tuple[torch.nn.Module | torch.Tensor, str]:
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]
    return module, tensor_name


def get_sdnq_params(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    sdnq_params = {}
    for name in sdnq_keys:
        sdnq_params[name] = getattr(module, name, None)
    return sdnq_params


def check_param_name_in(param_name: str, param_list: list[str]) -> str:
    split_param_name = param_name.split(".")
    for param in param_list:
        if param.startswith("."):
            if param_name.startswith(param[1:]):
                return param
            else:
                continue
        if (
            param_name == param
            or param in split_param_name
            or ("*" in param and re.match(param.replace(".*", "\\.*").replace("*", ".*"), param_name))
        ):
            return param
    return None


def check_quant_is_allowed(layer_class_name: str, weight: torch.Tensor, quantization_config, pre_quantized: bool = False) -> bool:
    if (
        layer_class_name in allowed_types
        and weight.dtype in {torch.float64, torch.float32, torch.float16, torch.bfloat16}
        and not (layer_class_name in embedding_types and not quantization_config.quant_embedding)
        and not ((layer_class_name in conv_types or layer_class_name in conv_transpose_types) and not quantization_config.quant_conv)
    ):
        if pre_quantized:
            return True
        if layer_class_name in conv_types:
            channel_size = weight.shape[1]
        elif layer_class_name in conv_transpose_types:
            channel_size = weight.shape[0]
        else:
            channel_size = weight.shape[-1]
        if channel_size >= quantization_config.minimum_allowed_channel_size and weight.numel() >= quantization_config.minimum_allowed_numel:
            return True
    return False


def check_quantized_matmul_is_allowed(use_quantized_matmul: bool, output_channel_size: int, channel_size: int) -> bool:
    return bool(
        use_quantized_matmul
        and output_channel_size >= 32 and channel_size >= 32
        and output_channel_size % 16 == 0 and channel_size % 16 == 0
    )


def get_quant_args_from_config(quantization_config: dict) -> dict:
    from .quantizer import SDNQConfig
    if isinstance(quantization_config, SDNQConfig):
        quantization_config_dict = quantization_config.to_dict()
    else:
        quantization_config_dict = quantization_config.copy()
    quantization_config_dict.pop("is_integer", None)
    quantization_config_dict.pop("quant_method", None)
    quantization_config_dict.pop("quantization_device", None)
    quantization_config_dict.pop("return_device", None)
    quantization_config_dict.pop("non_blocking", None)
    quantization_config_dict.pop("add_skip_keys", None)
    quantization_config_dict.pop("use_dynamic_quantization", None)
    quantization_config_dict.pop("use_static_quantization", None)
    quantization_config_dict.pop("use_stochastic_rounding", None)
    quantization_config_dict.pop("use_grad_ckpt", None)
    quantization_config_dict.pop("is_training", None)
    quantization_config_dict.pop("sdnq_version", None)
    if quantization_config_dict.get("modules_quant_config", None) is not None:
        for key in quantization_config_dict["modules_quant_config"]:
            quantization_config_dict["modules_quant_config"][key] = get_quant_args_from_config(quantization_config_dict["modules_quant_config"][key])
    return quantization_config_dict


def get_minimum_dtype(weights_dtype: str, param_name: str, modules_dtype_dict: dict[str, list[str]]):
    if len(modules_dtype_dict.keys()) > 0:
        for key, value in modules_dtype_dict.items():
            if check_param_name_in(param_name, value) is not None:
                key = key.lower()
                if key.startswith("minimum") or key.endswith(("bit", "bits")):
                    minimum_bits_str = key.removeprefix("minimum").removeprefix("-").removeprefix("_").removesuffix("bits").removesuffix("bit").removesuffix("-").removesuffix("_")
                    if minimum_bits_str.startswith("uint"):
                        is_unsigned = True
                        minimum_bits_str = minimum_bits_str.removeprefix("uint")
                    else:
                        is_unsigned = False
                        minimum_bits_str = minimum_bits_str.removeprefix("int")
                    minimum_bits = int(minimum_bits_str)
                    if dtype_dict[weights_dtype]["num_bits"] < minimum_bits:
                        if is_unsigned or minimum_bits <= 4:
                            return "uint" + minimum_bits_str
                        else:
                            return "int" + minimum_bits_str
                else:
                    return key
    return weights_dtype


def get_quant_kwargs(layer: torch.nn.Module, quantization_config, torch_dtype: torch.dtype | None = None, param_name: str = "", **kwargs) -> dict:
    from .quantizer import SDNQConfig
    if not isinstance(quantization_config, SDNQConfig):
        quantization_config = SDNQConfig(**quantization_config)
    layer_class_name = layer.__class__.__name__

    quant_kwargs = {
        "weights_dtype": quantization_config.weights_dtype,
        "scale_dtype": quantization_config.scale_dtype,
        "zero_point_dtype": quantization_config.zero_point_dtype,
        "quantized_matmul_dtype": quantization_config.quantized_matmul_dtype,
        "hadamard_group_size": quantization_config.hadamard_group_size,
        "group_size": quantization_config.group_size,
        "svd_rank": quantization_config.svd_rank,
        "svd_steps": quantization_config.svd_steps,
        "codebook_steps": quantization_config.codebook_steps,
        "dynamic_loss_threshold": quantization_config.dynamic_loss_threshold,
        "dynamic_loss_average_error_weight": quantization_config.dynamic_loss_average_error_weight,
        "dynamic_loss_max_row_error_weight": quantization_config.dynamic_loss_max_row_error_weight,
        "dynamic_loss_max_error_weight": quantization_config.dynamic_loss_max_error_weight,
        "weights_dtype_order": quantization_config.weights_dtype_order,
        "use_svd": quantization_config.use_svd,
        "use_hadamard": quantization_config.use_hadamard,
        "use_codebook": quantization_config.use_codebook,
        "use_codebook_scale": quantization_config.use_codebook_scale,
        "use_quantized_matmul": quantization_config.use_quantized_matmul,
        "use_quantized_matmul_conv": quantization_config.use_quantized_matmul_conv,
        "use_dynamic_quantization": quantization_config.use_dynamic_quantization,
        "use_stochastic_rounding": quantization_config.use_stochastic_rounding,
        "dequantize_fp32": quantization_config.dequantize_fp32,
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

    if layer_class_name in conv_transpose_types or layer_class_name in conv_types:
        quant_kwargs["use_quantized_matmul"] = quant_kwargs.pop("use_quantized_matmul_conv")
    else:
        quant_kwargs.pop("use_quantized_matmul_conv")

    if not quant_kwargs["use_dynamic_quantization"]:
        quant_kwargs.pop("dynamic_loss_threshold")
        quant_kwargs.pop("dynamic_loss_average_error_weight")
        quant_kwargs.pop("dynamic_loss_max_row_error_weight")
        quant_kwargs.pop("dynamic_loss_max_error_weight")
        quant_kwargs.pop("weights_dtype_order")

    quant_kwargs["weights_dtype"] = get_minimum_dtype(quant_kwargs["weights_dtype"], quant_kwargs["param_name"], quantization_config.modules_dtype_dict)
    if check_param_name_in(quant_kwargs["param_name"], quantization_config.modules_to_not_use_matmul) is not None:
        quant_kwargs["use_quantized_matmul"] = False

    return quant_kwargs


def get_quantized_matmul_dtype(weights_dtype: str, quantized_matmul_dtype: str | None = None) -> str:
    if quantized_matmul_dtype is None:
        if dtype_dict[weights_dtype]["is_integer"]:
            if weights_dtype == "uint8":
                quantized_matmul_dtype = "uint8"
            else:
                quantized_matmul_dtype = "int8"
        elif dtype_dict[weights_dtype]["num_bits"] < 16:
            quantized_matmul_dtype = "float8_e4m3fn"
        else:
            quantized_matmul_dtype = "float16"
    return quantized_matmul_dtype


def add_module_skip_keys(model: torch.nn.Module, quantization_config):
    if getattr(model, "_keep_in_fp32_modules", None) is not None:
        quantization_config.modules_to_not_convert.extend(model._keep_in_fp32_modules)
    if getattr(model, "_tied_weights_keys", None) is not None:
        if isinstance(model._tied_weights_keys, dict):
            quantization_config.modules_to_not_convert.extend(model._tied_weights_keys.keys())
            quantization_config.modules_to_not_convert.extend(model._tied_weights_keys.values())
        else:
            quantization_config.modules_to_not_convert.extend(model._tied_weights_keys)

    skip_key_list = module_skip_keys_dict.get(model.__class__.__name__, None)
    if skip_key_list is not None:
        quantization_config.modules_to_not_convert.extend(skip_key_list[0])
        for key, value in skip_key_list[1].items():
            if key in quantization_config.modules_dtype_dict:
                quantization_config.modules_dtype_dict[key].extend(value)
            else:
                quantization_config.modules_dtype_dict[key] = value

        quantized_matmul_dtype = get_quantized_matmul_dtype(quantization_config.weights_dtype, quantization_config.quantized_matmul_dtype)
        quantization_config.modules_to_not_use_matmul.extend(skip_key_list[2].get(quantized_matmul_dtype, []))
    else:
        quantization_config.modules_to_not_convert.extend(common_skip_keys)
        if getattr(model, "_skip_layerwise_casting_patterns", None) is not None:
            quantization_config.modules_to_not_convert.extend(model._skip_layerwise_casting_patterns)

    # dedupe
    quantization_config.modules_to_not_convert = list(set(quantization_config.modules_to_not_convert))
    quantization_config.modules_to_not_use_matmul = list(set(quantization_config.modules_to_not_use_matmul))
    for key, value in quantization_config.modules_dtype_dict.items():
        quantization_config.modules_dtype_dict[key] = list(set(value))

    return model, quantization_config
