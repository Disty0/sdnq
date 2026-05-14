import re
import torch

from .common import dtype_dict, common_skip_keys, module_skip_keys_dict, linear_types, conv_types, is_fp8_mm_supported


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
        for key in quantization_config_dict["modules_quant_config"].keys():
            quantization_config_dict["modules_quant_config"][key] = get_quant_args_from_config(quantization_config_dict["modules_quant_config"][key])
    return quantization_config_dict


def get_minimum_dtype(weights_dtype: str, param_name: str, modules_dtype_dict: dict[str, list[str]]):
    if len(modules_dtype_dict.keys()) > 0:
        for key, value in modules_dtype_dict.items():
            if check_param_name_in(param_name, value) is not None:
                key = key.lower()
                if key.startswith("minimum") or key.endswith("bit") or key.endswith("bits"):
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


def get_quant_kwargs(quant_kwargs: dict, modules_quant_config: dict[str, dict]) -> dict:
    param_key = check_param_name_in(quant_kwargs["param_name"], modules_quant_config.keys())
    if param_key is not None:
        for key, value in modules_quant_config[param_key].items():
            quant_kwargs[key] = value
    quant_kwargs["weights_dtype"] = get_minimum_dtype(quant_kwargs["weights_dtype"], quant_kwargs["param_name"], quant_kwargs["modules_dtype_dict"])
    return quant_kwargs


def update_modules_quant_config(quant_kwargs: dict, modules_quant_config: dict[str, dict], layer: torch.nn.Module) -> dict[str, dict]:
    layer_class_name = layer.__class__.__name__
    if layer_class_name in conv_types:
        use_quantized_matmul_key = "use_quantized_matmul_conv"
    else:
        use_quantized_matmul_key = "use_quantized_matmul"
    if (
        hasattr(layer, "sdnq_dequantizer")
        and (layer_class_name in linear_types or layer_class_name in conv_types)
        and quant_kwargs["use_dynamic_quantization"] and quant_kwargs[use_quantized_matmul_key]
        and quant_kwargs["quantized_matmul_dtype"] is None and not is_fp8_mm_supported
        and not dtype_dict[layer.sdnq_dequantizer.weights_dtype]["is_integer"] and dtype_dict[layer.sdnq_dequantizer.weights_dtype]["num_bits"] < 16
        and not layer.sdnq_dequantizer.use_quantized_matmul
    ):
        if quant_kwargs["param_name"] not in modules_quant_config.keys():
            modules_quant_config[quant_kwargs["param_name"]] = {}
        modules_quant_config[quant_kwargs["param_name"]][use_quantized_matmul_key] = False
    return modules_quant_config


def add_module_skip_keys(model, modules_to_not_convert: list[str] | None = None, modules_dtype_dict: dict[str, list[str]] | None = None):
    if modules_to_not_convert is None:
        modules_to_not_convert = []
    if modules_dtype_dict is None:
        modules_dtype_dict = {}
    if getattr(model, "_keep_in_fp32_modules", None) is not None:
        modules_to_not_convert.extend(model._keep_in_fp32_modules) # pylint: disable=protected-access
    if getattr(model, "_tied_weights_keys", None) is not None:
        if isinstance(model._tied_weights_keys, dict): # pylint: disable=protected-access
            modules_to_not_convert.extend(model._tied_weights_keys.keys()) # pylint: disable=protected-access
            modules_to_not_convert.extend(model._tied_weights_keys.values()) # pylint: disable=protected-access
        else:
            modules_to_not_convert.extend(model._tied_weights_keys) # pylint: disable=protected-access

    skip_key_list = module_skip_keys_dict.get(model.__class__.__name__, None)
    if skip_key_list is not None:
        modules_to_not_convert.extend(skip_key_list[0])
        for key, value in skip_key_list[1].items():
            if key in modules_dtype_dict.keys():
                modules_dtype_dict[key].extend(value)
            else:
                modules_dtype_dict[key] = value
    else:
        modules_to_not_convert.extend(common_skip_keys)
        if getattr(model, "_skip_layerwise_casting_patterns", None) is not None:
            modules_to_not_convert.extend(model._skip_layerwise_casting_patterns) # pylint: disable=protected-access

    # dedupe
    modules_to_not_convert = list(set(modules_to_not_convert))
    for key, value in modules_dtype_dict.items():
        modules_dtype_dict[key] = list(set(value))

    return model, modules_to_not_convert, modules_dtype_dict
