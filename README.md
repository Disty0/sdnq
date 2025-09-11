# SD.Next Quantization

For more info, please check out SD.Next SDNQ wiki page: https://github.com/vladmandic/sdnext/wiki/SDNQ-Quantization  


Example quantization config code for Diffusers and Transformers libraries:  

```py
from sdnq import SDNQQuantizer, SDNQConfig

import diffusers
diffusers.quantizers.auto.AUTO_QUANTIZER_MAPPING["sdnq"] = SDNQQuantizer
diffusers.quantizers.auto.AUTO_QUANTIZATION_CONFIG_MAPPING["sdnq"] = SDNQConfig

import transformers
transformers.quantizers.auto.AUTO_QUANTIZER_MAPPING["sdnq"] = SDNQQuantizer
transformers.quantizers.auto.AUTO_QUANTIZATION_CONFIG_MAPPING["sdnq"] = SDNQConfig

sdnq_config = SDNQConfig(
    weights_dtype="int8",
    group_size=0,
    quant_conv=False,
    use_quantized_matmul=True,
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

```py
from sdnq_training import apply_sdnq_to_module

model = apply_sdnq_to_module(
    model,
    weights_dtype="int8",
    use_grad_ckpt=True, # disable this if you are not using gradient checkpointing
    use_quantized_matmul=True,
    use_static_quantization=True, # quantize the model weights
    use_stochastic_quantization=True,
    modules_to_not_convert=["correction_coefs", "prediction_coefs", "lm_head", "embedding_projection"],
)
```


Example code for quantized optimizer states:  

```py
from sdnq_training import SDNQTensor

state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(p).add_(torch.finfo(p.dtype).eps), qtype="int8", sr=True)
```
