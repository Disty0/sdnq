# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

import torch

from ...common import compile_func, inference_context
from ...dequantizer import unpack_weights, dequantize_symmetric, dequantize_asymmetric, dequantize_codebook, dequantize_scales
from ...quant_utils import get_hadamard


@inference_context()
def quantized_embedding(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.FloatTensor,
    scale_2: torch.FloatTensor | None = None,
    scale_zero_point: torch.FloatTensor | None = None,
    zero_point: torch.FloatTensor | None = None,
    zero_point_scale: torch.FloatTensor | None = None,
    zero_point_2: torch.FloatTensor | None = None,
    svd_up: torch.FloatTensor | None = None,
    svd_down: torch.FloatTensor | None = None,
    hadamard: torch.FloatTensor | None = None,
    use_codebook: bool = False,
    use_codebook_scale: bool = False,
    embed_scale: torch.FloatTensor | float | None = None,
    result_dtype: torch.dtype | None = None,
    weight_shape: torch.Size | None = None,
    quantized_weight_shape: torch.Size | None = None,
    quantized_scale_shape: torch.Size | None = None,
    quantized_zero_point_shape: torch.Size | None = None,
    weights_dtype: str | None = None,
    scale_dtype: str | None = None,
    zero_point_dtype: str | None = None,
) -> torch.FloatTensor:
    return_shape = list(input.shape) + [weight_shape[-1] if weight_shape is not None else quantized_weight_shape[-1] if quantized_weight_shape is not None else weight.shape[-1]]
    input = input.flatten()

    weight = unpack_weights(
        weight, weights_dtype,
        quantized_weight_shape=quantized_weight_shape,
        dtype=torch.int32,
    )[input]
    scale = unpack_weights(
        scale, scale_dtype,
        quantized_weight_shape=quantized_scale_shape,
        dtype=scale_2.dtype if scale_2 is not None else torch.float32,
    )[input]
    if zero_point is not None:
        zero_point = unpack_weights(
            zero_point, zero_point_dtype,
            quantized_weight_shape=quantized_zero_point_shape,
            dtype=zero_point_scale.dtype if zero_point_scale is not None else torch.float32,
        )[input]

    scale, zero_point = dequantize_scales(
        None, scale,
        scale_2=scale_2,
        scale_zero_point=scale_zero_point,
        zero_point=zero_point,
        zero_point_scale=zero_point_scale,
        zero_point_2=zero_point_2,
        use_codebook=use_codebook_scale,
    )

    if use_codebook:
        result = dequantize_codebook(
            weight, scale,
            svd_up=svd_up[input] if svd_up is not None else svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            dtype=result_dtype,
        )
    elif zero_point is not None:
        result = dequantize_asymmetric(
            weight, scale, zero_point,
            svd_up=svd_up[input] if svd_up is not None else svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            dtype=result_dtype,
            )
    else:
        result = dequantize_symmetric(
            weight, scale,
            svd_up=svd_up[input] if svd_up is not None else svd_up,
            svd_down=svd_down,
            hadamard=hadamard,
            dtype=result_dtype,
        )
    del input

    result = result.view(return_shape).contiguous()
    if embed_scale is not None:
        result = result.mul_(embed_scale)

    return result


@inference_context()
def quantized_embedding_forward(self: torch.nn.Module, input: torch.Tensor) -> torch.FloatTensor:
    if self.sdnq_dequantizer.use_hadamard:
        hadamard = get_hadamard(self.sdnq_dequantizer.hadamard_group_size, dtype=self.sdnq_dequantizer.result_dtype, device=input.device)
    else:
        hadamard = None

    return quantized_embedding(
        input,
        self.weight,
        self.scale,
        scale_2=self.scale_2,
        scale_zero_point=self.scale_zero_point,
        zero_point=self.zero_point,
        zero_point_scale=self.zero_point_scale,
        zero_point_2=self.zero_point_2,
        svd_up=self.svd_up,
        svd_down=self.svd_down,
        hadamard=hadamard,
        use_codebook=self.sdnq_dequantizer.use_codebook,
        use_codebook_scale=self.sdnq_dequantizer.use_codebook_scale,
        embed_scale=getattr(self, "scalar_embed_scale", None),
        result_dtype=self.sdnq_dequantizer.result_dtype,
        weight_shape=self.sdnq_dequantizer.result_shape,
        quantized_weight_shape=self.sdnq_dequantizer.quantized_weight_shape,
        quantized_scale_shape=self.sdnq_dequantizer.quantized_scale_shape,
        quantized_zero_point_shape=self.sdnq_dequantizer.quantized_zero_point_shape,
        weights_dtype=self.sdnq_dequantizer.weights_dtype,
        scale_dtype=self.sdnq_dequantizer.scale_dtype,
        zero_point_dtype=self.sdnq_dequantizer.zero_point_dtype,
    )


quantized_embedding = compile_func(quantized_embedding)
