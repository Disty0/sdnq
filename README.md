# SDNQ: SD.Next Quantization Engine

SD.Next Quantization provides full cross-platform quantization to reduce memory usage and increase performance for any device.  
For more info, please check out SD.Next SDNQ wiki page: https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization  

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

```py
from sdnq import SDNQConfig
from sdnq.common import use_torch_compile as triton_is_available

sdnq_config = SDNQConfig(
    weights_dtype="int8", # Check out `sdnq.common.accepted_weight_dtypes` for all the supported dtypes.
    quantized_matmul_dtype=None, # overrides the quantized matmul dtype to be different than weights_dtype format.  
    group_size=0, # 0 means auto, -1 means disabled
    svd_rank=32,
    svd_steps=8,
    dynamic_loss_threshold=1e-2,
    use_svd=False,
    quant_conv=False,
    use_quantized_matmul=triton_is_available,
    use_quantized_matmul_conv=False,
    use_dynamic_quantization=False,
    dequantize_fp32=True,
    non_blocking=False,
    add_skip_keys=True,
    quantization_device="cuda",
    return_device="cuda",
    modules_to_not_convert=["correction_coefs", "prediction_coefs", "lm_head", "embedding_projection"],
    modules_dtype_dict={"int8": ["lm_head"]},
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


### Example code for quantized training:  
Note:  
 - Safetensors serialization is not supported with SDNQ training.  
   Either don't use Safetensors serialization or convert the quantized model to standard SDNQ model before saving.  
   You can also use `scripts/dequantize_sdnq_training.py` to dequantize an SDNQ Training model saved to the disk.  

```py
from sdnq.training import sdnq_training_post_load_quant
from sdnq.common import use_torch_compile as triton_is_available

quantized_model = sdnq_training_post_load_quant(
    model,
    weights_dtype="uint8",
    quantized_matmul_dtype="int8",
    group_size=32, # 0 means auto, -1 means disabled
    svd_rank=32,
    svd_steps=8,
    use_svd=False,
    use_grad_ckpt=True, # disable this if you are not using gradient checkpointing
    use_quantized_matmul=triton_is_available,
    use_static_quantization=True, # quantize the model weights
    use_stochastic_rounding=True,
    dequantize_fp32=True,
    non_blocking=False,
    add_skip_keys=True,
    quantization_device="cuda",
    return_device="cuda",
    modules_to_not_convert=["correction_coefs", "prediction_coefs", "lm_head", "embedding_projection"],
    modules_dtype_dict={"int8": ["lm_head"]},
)
```

### Example code for converting standard SDNQ model to training SDNQ Model:  

```py
from sdnq.training import convert_sdnq_model_to_training
from sdnq.common import use_torch_compile as triton_is_available
quantized_model = convert_sdnq_model_to_training(
    quantized_model,
    quantized_matmul_dtype="int8",
    use_grad_ckpt=True,
    use_quantized_matmul=triton_is_available,
    use_stochastic_rounding=True,
    dequantize_fp32=True,
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
    quantized_buffers_group_size=32,
    quantized_buffers_svd_rank=32,
    final_norm_mode="clip", # can be one of ["none", "clip", "rms", "rms_clip", "relative", "muon"]
    use_kahan=False,
    use_cautious=False,
    use_stochastic_rounding=True,
    use_stochastic_buffers=True,
    use_svd_quantization=False,
    use_torch_compile=False,
    offload_buffers=False,
    offload_non_blocking=True,
)
```


### Example code for quantized optimizer states for custom optimizers or Tensors:  

```py
from sdnq.training import SDNQTensor

state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(p), weights_dtype="uint8", group_size=32, use_stochastic_rounding=True)
```
