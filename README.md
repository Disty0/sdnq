# SDNQ: SD.Next Quantization Engine

SD.Next Quantization provides full cross-platform quantization to reduce memory usage and increase performance for any device.  
- SDNQ is written fully in PyTorch and can be compiled with torch.compile into different backends such as Inductor and OpenVINO.  
- SDNQ can run on any device (MPS (Apple Mac), NPU, CPU, ARM, Android etc.) with PyTorch Eager fallback mode.  
  - CUDA (Nvidia GPU), ROCm (AMD GPU), XPU (Intel GPU) and CPU devices utilizes the faster Inductor backend by default if Triton or Inductor is available.  
- SDNQ supports every quantization type from 1 bit to 16 bits including int, uint, fp and ufp types totaling to 176 storage types for inference and training.  
- SDNQ supports Hadamard Rotations and SVD Quantization on both quantized weights and quantized matmul for inference and training.  
- SDNQ supports INT8, FP8 and FP16 quantized matmul on supported Nvidia, AMD and Intel GPUs for inference and training with any quantized weights type.  
- SDNQ supports fast INT8 quantized mamtul on any x86 or ARM CPUs and Intel NPUs via OpenVINO matmul (requires manual installation of OpenVINO via `pip install openvino`).  
- SDNQ supports full parameter quantized training with quantized weights and / or quantized matmul and also offers quantized optimizers for training.  
- SDNQ supports direct math to be done on the quantized model on training (aka supports updating the quantized model weights directy).  
- SDNQ also offers fast Quantized Attention kernels for Nvidia, AMD and Intel Arc GPUs with Triton.  

For more info, please see SD.Next SDNQ Wiki page: https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization  

### Install command:  
```sh
pip install sdnq
```


### Example code to load pre-quantized models:  
Pre-quantized models can be found here: https://huggingface.co/collections/Disty0/sdnq  

```py
import torch
from sdnq import SDNQConfig # import sdnq to register it into diffusers and transformers
pipe_or_quantized_model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16)
```

### Example code for enabling or disabling quantized matmul with a pre-quantized model:  
```py
from sdnq.loader import apply_sdnq_options_to_model
quantized_model = apply_sdnq_options_to_model(quantized_model, use_quantized_matmul=True)
```


### Example quantization config code for Diffusers and Transformers libraries:  

For more information about the options, see [SDNQ Wiki](https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization) and `SDNQConfig` docstring.  

```py
from sdnq import SDNQConfig
from sdnq.common import use_torch_compile as triton_is_available

sdnq_config = SDNQConfig(
    weights_dtype="int8", # see `sdnq.common.accepted_weight_dtypes` for all the supported dtypes.
    quantized_matmul_dtype=None, # overrides the quantized matmul dtype to be different than weights_dtype format.
    group_size=0, # 0 means auto, -1 means disabled (aka. uses row-wise quant)
    hadamard_group_size=256,
    svd_rank=32,
    svd_steps=8,
    dynamic_loss_threshold=None, # None or negative number means auto select based on weights_dtype
    use_svd=False,
    use_hadamard=False,
    quant_conv=False,
    quant_embedding=False,
    use_quantized_matmul=triton_is_available, # use quantized matmul (False means no quantized matmul at all)
    use_quantized_matmul_conv=False,
    use_dynamic_quantization=False, # dynamically select a per layer quantization type based on the dynamic_loss_threshold
    dequantize_fp32=True, # keeps the quant scales in FP32 and compute the de-quant steps in FP32. Highly recommended to enable this option
    non_blocking=False,
    add_skip_keys=True,
    modules_to_not_convert=["correction_coefs", "prediction_coefs", "lm_head", "embedding_projection"],
    modules_to_not_use_matmul=["x_embedder"],
    modules_dtype_dict={"int8": ["lm_head"]},
    modules_quant_config={"embed_tokens_per_layer": {"quantization_device": "cpu"}},
    quantization_device="cuda",
    return_device="cuda",
)

quantized_model = AutoModel.from_pretrained(model_path, quantization_config=sdnq_config)
```

### Example code for saving a quantized Diffusers or Transformers model:  

```py
pipe_or_quantized_model.save_pretrained("path_to_save_the_quantized_model")
```


### Example quantization code for post load quantization on any model:  

```py
from sdnq import sdnq_post_load_quant

model = sdnq_post_load_quant(
    model,
    **kwargs_are_the_same_as_SDNQConfig,
)
```


### Example code for using SDNQ Attention as SDPA replacement for Inference:  
```py
from functools import wraps
from sdnq.kernels.triton_atten import sdnq_triton_atten

sdpa_pre_sdnq_atten = torch.nn.functional.scaled_dot_product_attention
@wraps(sdpa_pre_sdnq_atten)
def sdpa_sdnq_atten(query: torch.FloatTensor, key: torch.FloatTensor, value: torch.FloatTensor, attn_mask: torch.Tensor | None = None, dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None, enable_gqa: bool = False, **kwargs) -> torch.FloatTensor:
    if (
        not is_causal
        and query.device.type != "cpu"
        and key.device == query.device
        and value.device == query.device
        and (query.shape[-1] >= 32 and key.shape[-1] >= 32 and value.shape[-1] >= 32) # Dim < 32 is unsupported by Matrix Cores
        and (query.shape[-2] >= 512 or key.shape[-2] >= 512) # Skip TE
        and query.shape[-3] > 1 # Skip VAE
    ):
        return sdnq_triton_atten(
            query=query, key=key, value=value, attn_mask=attn_mask, scale=scale, enable_gqa=enable_gqa,
            matmul_dtype="int8", # can be one of "no", "int8", "float8_e4m3fn", "float16".
            pv_matmul_dtype="no", # can be one of "no", "int8", "float8_e4m3fn", "float16".
            smooth_k=False,
            use_hadamard=False,
            hadamard_group_size=256,
            do_quantize=True, # Set this to False to disable the quantized matmul usage
            out_dtype=None, # Set this to a torch.dtype like torch.float32 if you want the output dtype to be different than inputs
        )
    else:
        if enable_gqa:
            kwargs["enable_gqa"] = enable_gqa
        return sdpa_pre_sdnq_atten(query=query, key=key, value=value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)
torch.nn.functional.scaled_dot_product_attention = sdpa_sdnq_atten
```


### Example code for quantized training:  

For more information about the options, see [SDNQ Wiki](https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization) and `SDNQConfig` docstring.  
Note:  
 - Safetensors serialization is not supported with SDNQ training.  
   Either don't use Safetensors serialization or convert the quantized model to standard SDNQ model before saving.  
   You can also use `scripts/dequantize_sdnq_training.py` to dequantize an SDNQ Training model saved to the disk.  

```py
from sdnq.training import sdnq_training_post_load_quant
from sdnq.common import use_torch_compile as triton_is_available

quantized_model = sdnq_training_post_load_quant(
    model,
    weights_dtype="uint8", # Check out `sdnq.common.accepted_weight_dtypes` for all the supported dtypes.
    quantized_matmul_dtype=None, # overrides the quantized matmul dtype to be different than weights_dtype format.
    group_size=32, # 0 means auto, -1 means disabled (aka. uses row-wise quant)
    hadamard_group_size=256,
    svd_rank=32,
    svd_steps=8,
    use_svd=False,
    use_hadamard=False,
    use_grad_ckpt=True, # disable this if you are not using gradient checkpointing
    use_quantized_matmul=triton_is_available, # use quantized matmul on the forward pass and the backward pass (False means no quantized matmul at all)
    use_static_quantization=True, # quantize the model weights (False means model weights will be kept unquantized and only quantized matmul (if enabled) will be used)
    use_stochastic_rounding=True,
    dequantize_fp32=True, # keeps the quant scales in FP32 and compute the de-quant steps in FP32. Highly recommended to enable this option
    non_blocking=False,
    add_skip_keys=True,
    modules_to_not_convert=["correction_coefs", "prediction_coefs", "lm_head", "embedding_projection"],
    modules_to_not_use_matmul=["x_embedder"],
    modules_dtype_dict={"int8": ["lm_head"]},
    modules_quant_config={"embed_tokens_per_layer": {"quantization_device": "cpu"}},
    quantization_device="cuda",
    return_device="cuda",
)
```

### Example code for converting standard SDNQ model to training SDNQ Model:  

```py
from sdnq.training import convert_sdnq_model_to_training
from sdnq.common import use_torch_compile as triton_is_available
quantized_model = convert_sdnq_model_to_training(
    quantized_model,
    quantized_matmul_dtype=None, # overrides the quantized matmul dtype to be different than weights_dtype format.
    use_grad_ckpt=True, # disable this if you are not using gradient checkpointing
    use_quantized_matmul=triton_is_available, # use quantized matmul on the forward pass and the backward pass (False means no quantized matmul at all)
    use_stochastic_rounding=True,
    dequantize_fp32=True, # keeps the quant scales in FP32 and compute the de-quant steps in FP32. Highly recommended to enable this option
)
```

### Example code for converting training SDNQ model to standard SDNQ Model:  

```py
from sdnq.training import convert_training_model_to_sdnq
quantized_model = convert_training_model_to_sdnq(quantized_model)
```


### Example code for quantized optimizer states:  
```py
from sdnq.optim import Adafactor, AdamW, CAME, Lion, Muon
optimizer = AdamW(
    parameters,
    use_quantized_buffers=True,
    quantized_buffers_dtype="uint8",
    quantized_buffers_hadamard_group_size=256,
    quantized_buffers_group_size=32,
    quantized_buffers_svd_rank=32,
    final_norm_mode="clip", # can be one of ["none", "clip", "rms", "rms_clip", "relative", "muon"]
    use_kahan=False,
    use_cautious=False,
    use_stochastic_rounding=True,
    use_stochastic_buffers=True,
    quantized_buffers_use_svd=False,
    quantized_buffers_use_hadamard=False,
    use_torch_compile=False,
    offload_buffers=False,
    offload_non_blocking=True,
)
```


### Example code for quantized optimizer states for custom optimizers or Tensors:  

```py
from sdnq.training import SDNQTensor

state["exp_avg"] = SDNQTensor.from_float(
    torch.zeros_like(p),
    weights_dtype="int8",
    hadamard_group_size=256,
    group_size=32,
    svd_rank=32,
    svd_steps=8,
    use_svd=False,
    use_hadamard=False,
    use_stochastic_rounding=True,
    dequantize_fp32=True, # keeps the quant scales in FP32 and compute the de-quant steps in FP32. Highly recommended to enable this option
    layer_class_name=None, # can be "Linear", "Conv2d" etc.
)
```


### Environment Variables

- **SDNQ_USE_TORCH_COMPILE**: Overrides the default Triton and torch.compile test done by SDNQ.  
  Can be `0` or `1`. Default is None (auto-detect)  
- **SDNQ_ALLOW_FP8_MM**: Overrides the default FP8 matmul support test done by SDNQ.  
  This option is used with the `use_dynamic_quantization` option and within the `apply_sdnq_options_to_module` function.  
  Can be `0` or `1`. Default is None (auto-detect)  
- **SDNQ_USE_TENSORWISE_FP8_MM**: Force the use of software row-wise quantization via tensorwise kernels on unsupported hardware.  
  Can be `0` or `1`. Default is None (auto-detect)  
- **SDNQ_USE_CONTIGUOUS_MM**: Force the use of contiguous matmul instead of regular transposed matmul.  
  Some devices can perform much better with contiguous matmul.  
  Can be `0` or `1`. Default is None (auto-detect)  
- **SDNQ_USE_TRITON_MM**: Force the use of Triton MM kernels for INT8 MM instead of torch._int_mm.  
  AMD RDNA2 GPUs requires Triton MM kernels for INT8 MM support.  
  Triton MM kernels can outperform torch._int_mm on Intel and AMD GPUs.  
  Can be `0` or `1`. Default is None (auto-detect)  
- **SDNQ_USE_OPENVINO_MM**: Force the use of OpenVINO MM kernels for INT8 MM instead of torch._int_mm.  
  OpenVINO MM kernels will outperform torch._int_mm on CPUs.  
  Requires manual installation of OpenVINO via `pip install openvino`.  
  Can be `0` or `1`. Default is None (auto-detect)  
- **SDNQ_OPENVINO_DEVICE**: Overrides the default OpenVINO device used for OpenVINO MM.  
  Must be name of an OpenVINO device such as `CPU`. Default is `HETERO:NPU,CPU` if `NPU` is available else `CPU`  
- **SDNQ_COMPILE_KWARGS**: A dict of kwargs to override the kwargs used on torch.compile for SDNQ.  
  `SDNQ_COMPILE_KWARGS` is an advanced option, don't touch this if you don't know exactly what you are doing.  
  Must be json string such as `{"fullgraph": true}`. Default is None (auto-detect)  
- **SDNQ_DEVICE**: A device to override the default SDNQ device detection.  
  Must be name of a torch.device such as `mps`. Default is None (auto-detect)  
- **SDNQ_DTYPE**: A dtype to override the default SDNQ dtype detection based on the detected device.  
  Must be name of a torch.dtype such as `bfloat16`. Default is None (auto-detect)  
