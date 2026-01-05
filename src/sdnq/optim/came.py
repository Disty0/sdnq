from typing import Tuple, Optional, Iterator

import torch

from ..training import SDNQTensor

from .optimizer import SDNQOptimizer
from .utils import lerp_buffer_stochastic_, apply_norm_to_update_
from .adafactor import approx_sq_grad


class CAME(SDNQOptimizer):
    _extra_group_keys = {"norm_mode"}
    _keep_in_fp32_keys = {"exp_avg_sq", "exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_res_row", "exp_avg_res_col"}
    _group_keys = set.union(SDNQOptimizer._base_group_keys, _extra_group_keys)

    def __init__(self, params, **kwargs):
        if isinstance(params, (torch.nn.Parameter, Iterator)) or (isinstance(params, (list, tuple)) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            group["betas"] = self.get_default_kwarg(group, kwargs, "betas", (0.9, 0.95, 0.99))
            group["norm_mode"] = self.get_default_kwarg(group, kwargs, "norm_mode", "rms_clip")
            group = self.apply_group_defaults(group, **kwargs)
            assert set(group.keys()) == self._group_keys
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def init_state(self, param: torch.Tensor, group: dict, state: dict) -> dict:
        grad_shape = param.grad.shape
        if len(grad_shape) >= 2:
            state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1], dtype=torch.float32, device=param.device)
            state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=torch.float32, device=param.device)
            state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1], dtype=torch.float32, device=param.device)
            state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=torch.float32, device=param.device)
        else:
            state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
        if group["use_quantized_buffers"]:
            state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), weights_dtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], use_stochastic_rounding=group["use_stochastic_buffers"])
        else:
            state["exp_avg"] = torch.zeros_like(param)
        return state

    @torch.no_grad()
    def get_param_update(self, param_fp32: torch.FloatTensor, grad: torch.FloatTensor, group: dict, state: dict) -> torch.FloatTensor:
        return came_update(
            grad=grad,
            param=param_fp32,
            exp_avg_sq_row=state.get("exp_avg_sq_row", None),
            exp_avg_sq_col=state.get("exp_avg_sq_col", None),
            exp_avg_res_row=state.get("exp_avg_res_row", None),
            exp_avg_res_col=state.get("exp_avg_res_col", None),
            exp_avg_sq=state.get("exp_avg_sq", None),
            exp_avg=state["exp_avg"],
            step=state["step"],
            betas=group["betas"],
            clips=group["clip_threshold"],
            norm_mode=group["norm_mode"],
            use_stochastic_buffers=group["use_stochastic_buffers"],
        )


def came_update(
    grad: torch.FloatTensor,
    param: torch.FloatTensor,
    exp_avg_sq_row: torch.FloatTensor,
    exp_avg_sq_col: torch.FloatTensor,
    exp_avg_res_row: torch.FloatTensor,
    exp_avg_res_col: torch.FloatTensor,
    exp_avg_sq: Optional[torch.FloatTensor],
    exp_avg: torch.FloatTensor,
    step: int,
    betas: Tuple[float, float, float],
    clips: Tuple[float],
    norm_mode: str = "rms_clip",
    use_stochastic_buffers: bool = False,
) -> torch.FloatTensor:
    beta1, beta2, beta3 = betas
    clip = clips[0]

    one_minus_beta2 = 1 - beta2
    update = torch.square(grad)
    if exp_avg_sq is None:
        exp_avg_sq_row.lerp_(update.mean(dim=-1), one_minus_beta2)
        exp_avg_sq_col.lerp_(update.mean(dim=-2), one_minus_beta2)
        update = approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
    else:
        exp_avg_sq.lerp_(update, one_minus_beta2)
        update = exp_avg_sq.rsqrt()

    update = update.mul_(grad).nan_to_num_().clamp_(-clip,clip)
    update = apply_norm_to_update_(update, param, norm_mode, clips)

    exp_avg, exp_avg_fp32 = lerp_buffer_stochastic_(exp_avg, update, 1 - beta1, use_stochastic_rounding=use_stochastic_buffers)
    if exp_avg_sq is None:
        res = torch.sub(update, exp_avg_fp32).square_()
        one_minus_beta3 = 1 - beta3
        exp_avg_res_row.lerp_(res.mean(dim=-1), one_minus_beta3)
        exp_avg_res_col.lerp_(res.mean(dim=-2), one_minus_beta3)
        update = approx_sq_grad(exp_avg_res_row, exp_avg_res_col).mul_(exp_avg_fp32)
    else:
        update = exp_avg_fp32.clone()
    del exp_avg_fp32

    update = update.nan_to_num_().clamp_(-clip,clip)
    return update
