# SDNQ: SD.Next Quantization Engine

For more info, please check out SD.Next SDNQ wiki page: https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization  

Install command:
```sh
pip install git+https://github.com/Disty0/sdnq
```


Example quantization config code for Diffusers and Transformers libraries:  

```py
from sdnq import SDNQConfig
from sdnq.common import use_torch_compile as triton_is_available

sdnq_config = SDNQConfig(
    weights_dtype="int8",
    group_size=0,
    svd_rank=128,
    use_svd=False,
    quant_conv=False,
    use_quantized_matmul=triton_is_available,
    use_quantized_matmul_conv=False,
    dequantize_fp32=False,
    non_blocking=False,
    quantization_device="cuda",
    return_device="cuda",
    modules_to_not_convert=["correction_coefs", "prediction_coefs", "lm_head", "embedding_projection"],
    modules_dtype_dict={"int8": ["lm_head"]},
)

model = AutoModel.from_pretrained(model_path, quantization_config=sdnq_config)
```


Example code for quantized training:  
Note:  
 - Only INT8, UINT8 and FP8 (E4) are supported in training.  
 - Safetensors serialization is not supported with static quantized training.  

```py
from sdnq.training import apply_sdnq_to_module
from sdnq.common import use_torch_compile as triton_is_available

model = apply_sdnq_to_module(
    model,
    weights_dtype="uint8",
    quantized_matmul_dtype="int8",
    group_size=32, # -1 means disabled
    use_grad_ckpt=True, # disable this if you are not using gradient checkpointing
    use_quantized_matmul=triton_is_available,
    use_static_quantization=True, # quantize the model weights
    use_stochastic_quantization=True,
    modules_to_not_convert=["correction_coefs", "prediction_coefs", "lm_head", "embedding_projection"],
)
```


Example code for quantized optimizer states:  
```py
from sdnq.optim import Adafactor, AdamW, CAME, Lion, Muon
optimizer = AdamW(
    parameters,
    bf16_stochastic_round=False, # for BF16 weights. Do not enable this with static quantized weights
    use_quantized_buffers=True,
    quantized_buffers_dtype="uint8",
    quantized_buffers_group_size=32,
    use_stochastic_quantization=True,
)
```


Example code for quantized optimizer states for custom optimizers:  

```py
from sdnq.training import SDNQTensor

state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(p), qtype="uint8", group_size=32, sr=True)
```
