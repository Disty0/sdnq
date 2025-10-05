import os
import json
import torch
from safetensors import safe_open
from diffusers.models.modeling_utils import ModelMixin

from .quantizer import SDNQConfig, apply_sdnq_to_module
from .common import use_contiguous_mm, use_tensorwise_fp8_matmul
from .dequantizer import dequantize_symmetric_compiled, quantize_fp8


def save_sdnq_model(model: ModelMixin, sdnq_config: SDNQConfig, model_path: str) -> None:
    model.save_pretrained(model_path)
    sdnq_config.to_json_file(os.path.join(model_path, "quantization_config.json"))


def load_sdnq_model(model_cls: ModelMixin, model_path: str, use_quantized_matmul: bool = False) -> ModelMixin:
    with torch.device("meta"):
        with open(os.path.join(model_path, "quantization_config.json"), "r") as f:
            quantization_config = json.load(f)
        quantization_config.pop("is_integer", None)
        quantization_config.pop("quant_method", None)
        quantization_config.pop("quantization_device", None)
        quantization_config.pop("return_device", None)
        quantization_config.pop("non_blocking", None)
        config = model_cls.load_config(model_path)
        model = model_cls.from_config(config)
        model = apply_sdnq_to_module(model, **quantization_config)

    state_dict = {}
    with safe_open(os.path.join(model_path, "diffusion_pytorch_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
    model.load_state_dict(state_dict, assign=True)
    del state_dict
    if use_quantized_matmul and not quantization_config["use_quantized_matmul"]:
        model = enable_quantized_mamtul(model)
    return model


def enable_quantized_mamtul(model):
    has_children = list(model.children())
    if not has_children:
        return model
    for param_name, module in model.named_children():
        if hasattr(module, "sdnq_dequantizer"):
            if not module.sdnq_dequantizer.use_quantized_matmul:
                if module.sdnq_dequantizer.weights_dtype in {"int8", "float8_e4m3fn"} and module.sdnq_dequantizer.result_shape != module.weight.shape:
                    if module.sdnq_dequantizer.weights_dtype == "int8":
                        module.weight.data, module.scale.data = module.sdnq_dequantizer.re_quantize_matmul(module.weight, module.scale, module.zero_point, None, None)
                    elif module.sdnq_dequantizer.weights_dtype == "float8_e4m3fn":
                        module.weight.data, module.scale.data = quantize_fp8(dequantize_symmetric_compiled(module.weight, module.scale, module.sdnq_dequantizer.result_dtype, module.sdnq_dequantizer.result_shape))
                        module.weight.data, module.scale.data = module.weight.t_(), module.scale.t_()
                        if not use_tensorwise_fp8_matmul:
                            module.scale.data = module.scale.to(dtype=torch.float32)
                    if use_contiguous_mm:
                        module.weight.data = module.weight.contiguous()
                    elif module.weight.is_contiguous():
                        module.weight.data = module.weight.t_().contiguous().t_()
                if module.svd_up is not None:
                    module.svd_up.data = module.svd_up.t_()
                    module.svd_down.data = module.svd_down.t_()
                module.sdnq_dequantizer.use_quantized_matmul = True
        module = enable_quantized_mamtul(module)
    return model
