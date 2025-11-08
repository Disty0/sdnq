from typing import Tuple, Optional, Iterator

import torch

from .optimizer import SDNQOptimizer, apply_norm_to_update_
from .stochastic import copy_stochastic_
from sdnq.training import SDNQTensor


class CAME(SDNQOptimizer):
    _extra_group_keys = {"norm_mode"}
    _keep_in_fp32_keys = {"exp_avg_sq", "exp_avg_sq_row", "exp_avg_sq_col", "exp_avg_res_row", "exp_avg_res_col"}
    _group_keys = set.union(SDNQOptimizer._base_group_keys, _extra_group_keys)

    def __init__(self, params, **kwargs):
        if isinstance(params, (torch.nn.Parameter, Iterator)) or (isinstance(params, list) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            group["betas"] = group.get("betas", (0.9, 0.95, 0.99))
            group["norm_mode"] = group.get("norm_mode", "rms_clip")
            group = self.apply_group_defaults(group)
            assert set(group.keys()) == self._group_keys
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        grad_scale = getattr(self, "grad_scale", None)
        found_inf = getattr(self, "found_inf", 0)

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None or found_inf > 0:
                    continue

                state = self.state[param]
                grad_shape = param.grad.shape
                factored = len(grad_shape) >= 2

                if len(state) == 0:
                    state["step"] = 0
                    if group["use_quantized_buffers"]:
                        state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], sr=group["use_stochastic_buffers"])
                    else:
                        state["exp_avg"] = torch.zeros_like(param)

                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1], dtype=torch.float32, device=param.device)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=torch.float32, device=param.device)
                        state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1], dtype=torch.float32, device=param.device)
                        state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=torch.float32, device=param.device)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)

                state["step"] += 1
                param_fp32 = param.to(dtype=torch.float32)
                grad = param.grad.to(dtype=torch.float32)
                if grad_scale is not None:
                    grad.div_(grad_scale.to(dtype=torch.float32))

                update = came_update(
                    grad=grad,
                    param=param_fp32,
                    exp_avg_sq_row=state["exp_avg_sq_row"] if factored else None,
                    exp_avg_sq_col=state["exp_avg_sq_col"] if factored else None,
                    exp_avg_res_row=state["exp_avg_res_row"] if factored else None,
                    exp_avg_res_col=state["exp_avg_res_col"] if factored else None,
                    exp_avg_sq=state["exp_avg_sq"] if not factored else None,
                    exp_avg=state["exp_avg"],
                    step=state["step"],
                    betas=group["betas"],
                    clips=group["clip_threshold"],
                    norm_mode=group["norm_mode"],
                    use_stochastic_buffers=group["use_stochastic_buffers"],
                ).to(dtype=torch.float32)

                self.update_param_(
                    param=param,
                    param_fp32=param_fp32,
                    grad=grad,
                    update=update,
                    learning_rate=group["lr"],
                    weight_decay=group["weight_decay"],
                    clips=group["clip_threshold"],
                    final_norm_mode=group["final_norm_mode"],
                    use_cautious=group["use_cautious"],
                    use_stochastic_rounding=group["use_stochastic_rounding"],
                )

        return loss


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

    if exp_avg.dtype != torch.float32:
        exp_avg_fp32 = exp_avg.to(dtype=torch.float32).lerp_(update, 1 - beta1)
        if use_stochastic_buffers and not isinstance(exp_avg, SDNQTensor):
            copy_stochastic_(exp_avg, exp_avg_fp32)
        else:
            exp_avg.copy_(exp_avg_fp32)
    else:
        exp_avg.lerp_(update, 1 - beta1)
        exp_avg_fp32 = exp_avg

    if exp_avg_sq is None:
        res = torch.sub(update, exp_avg_fp32).square_()
        one_minus_beta3 = 1 - beta3
        exp_avg_res_row.lerp_(res.mean(dim=-1), one_minus_beta3)
        exp_avg_res_col.lerp_(res.mean(dim=-2), one_minus_beta3)
        update = approx_sq_grad(exp_avg_res_row, exp_avg_res_col).mul_(exp_avg_fp32)
    else:
        update = exp_avg_fp32.clone()

    update = update.nan_to_num_().clamp_(-clip,clip)
    return update


def approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    r_factor = torch.div(exp_avg_sq_row, exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
    c_factor = exp_avg_sq_col.rsqrt().unsqueeze(-2)
    return torch.mul(r_factor, c_factor)
