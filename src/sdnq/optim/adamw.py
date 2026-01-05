from typing import Tuple, Iterator

import torch

from ..training import SDNQTensor

from .optimizer import SDNQOptimizer
from .utils import lerp_buffer_stochastic_


class AdamW(SDNQOptimizer):
    _extra_group_keys = {}
    _keep_in_fp32_keys = {}
    _group_keys = set.union(SDNQOptimizer._base_group_keys, _extra_group_keys)

    def __init__(self, params, **kwargs):
        if isinstance(params, (torch.nn.Parameter, Iterator)) or (isinstance(params, (list, tuple)) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            group = self.apply_group_defaults(group, **kwargs)
            assert set(group.keys()) == self._group_keys
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def init_state(self, param: torch.Tensor, group: dict, state: dict) -> dict:
        if group["use_quantized_buffers"]:
            state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), weights_dtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], use_stochastic_rounding=group["use_stochastic_buffers"])
            state["exp_avg_sq"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), weights_dtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], use_stochastic_rounding=group["use_stochastic_buffers"])
        else:
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
        return state

    @torch.no_grad()
    def get_param_update(self, param_fp32: torch.FloatTensor, grad: torch.FloatTensor, group: dict, state: dict) -> torch.FloatTensor:
        return adam_update(
            grad=grad,
            exp_avg=state["exp_avg"],
            exp_avg_sq=state["exp_avg_sq"],
            step=state["step"],
            betas=group["betas"],
            clip=group["clip_threshold"][0],
            use_stochastic_buffers=group["use_stochastic_buffers"],
        )


def adam_update(
    grad: torch.FloatTensor,
    exp_avg: torch.FloatTensor,
    exp_avg_sq: torch.FloatTensor,
    step: int,
    betas: Tuple[float, float],
    clip: float,
    use_stochastic_buffers: bool = False,
) -> torch.FloatTensor:
    beta1, beta2 = betas

    exp_avg, exp_avg_fp32 = lerp_buffer_stochastic_(exp_avg, grad, 1 - beta1, use_stochastic_rounding=use_stochastic_buffers)
    exp_avg_c = exp_avg_fp32 / (1 - beta1 ** step)
    del exp_avg_fp32

    exp_avg_sq, exp_avg_sq_fp32 = lerp_buffer_stochastic_(exp_avg_sq, grad.square(), 1 - beta2, use_stochastic_rounding=use_stochastic_buffers)
    exp_avg_sq_c = exp_avg_sq_fp32 / (1 - beta2 ** step)
    del exp_avg_sq_fp32

    return exp_avg_c.mul_(exp_avg_sq_c.rsqrt_()).nan_to_num_().clamp_(-clip,clip)
