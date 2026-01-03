from typing import Tuple, Optional, Iterator

import torch

from ..training import SDNQTensor

from .optimizer import SDNQOptimizer
from .utils import get_param_grad, update_param_, lerp_buffer_stochastic_, apply_norm_to_update_


class Adafactor(SDNQOptimizer):
    _extra_group_keys = {"use_first_moment", "norm_mode"}
    _keep_in_fp32_keys = {"variance", "row_var", "col_var"}
    _group_keys = set.union(SDNQOptimizer._base_group_keys, _extra_group_keys)

    def __init__(self, params, **kwargs):
        if isinstance(params, (torch.nn.Parameter, Iterator)) or (isinstance(params, (list, tuple)) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            group["lr"] = self.get_default_kwarg(group, kwargs, "lr", 1e-2)
            group["betas"] = self.get_default_kwarg(group, kwargs, "betas", (-0.8, 0.95))
            group["norm_mode"] = self.get_default_kwarg(group, kwargs, "norm_mode", "relative")
            group["use_first_moment"] = self.get_default_kwarg(group, kwargs, "use_first_moment", False)
            group = self.apply_group_defaults(group, **kwargs)
            assert set(group.keys()) == self._group_keys
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        grad_scale = getattr(self, "grad_scale", None)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                grad_shape = param.grad.shape
                factored = len(grad_shape) >= 2

                if len(state) == 0:
                    state["step"] = 0
                    if factored:
                        state["row_var"] = torch.zeros(grad_shape[:-1], dtype=torch.float32, device=param.device)
                        state["col_var"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=torch.float32, device=param.device)
                    else:
                        state["variance"] = torch.zeros_like(param, dtype=torch.float32)

                    if group["use_first_moment"]:
                        if group["use_quantized_buffers"]:
                            state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), weights_dtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], use_stochastic_rounding=group["use_stochastic_buffers"])
                        else:
                            state["exp_avg"] = torch.zeros_like(param)

                state["step"] += 1
                param_fp32, grad = get_param_grad(param, clip=group["clip_threshold"][0], grad_scale=grad_scale)

                if factored and group["offload_buffers"] and group["use_first_moment"]:
                    state["exp_avg"] = state["exp_avg"].to(param.device, non_blocking=group["offload_non_blocking"])

                update = adafactor_update(
                    param=param_fp32,
                    grad=grad,
                    row_var=state["row_var"] if factored else None,
                    col_var=state["col_var"] if factored else None,
                    variance=state["variance"] if not factored else None,
                    exp_avg=state["exp_avg"] if group["use_first_moment"] else None,
                    step=state["step"],
                    betas=group["betas"],
                    clips=group["clip_threshold"],
                    norm_mode=group["norm_mode"],
                    use_stochastic_buffers=group["use_stochastic_buffers"],
                ).to(dtype=torch.float32)

                if factored and group["offload_buffers"] and group["use_first_moment"]:
                    state["exp_avg"] = state["exp_avg"].to("cpu", non_blocking=False)

                update_param_(
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


def adafactor_update(
    param: torch.FloatTensor,
    grad: torch.FloatTensor,
    row_var: torch.FloatTensor,
    col_var: torch.FloatTensor,
    variance: Optional[torch.FloatTensor],
    exp_avg: Optional[torch.FloatTensor],
    step: int,
    betas: Tuple[float, float],
    clips: Tuple[float, float],
    norm_mode: str = "relative",
    use_stochastic_buffers: bool = False,
) -> torch.FloatTensor:
    clip = clips[0]
    beta1, beta2 = betas

    beta_t = step**beta1
    update = torch.square(grad)
    if variance is None:
        row_var.lerp_(update.mean(dim=-1), beta_t)
        col_var.lerp_(update.mean(dim=-2), beta_t)
        update = approx_sq_grad(row_var, col_var)
    else:
        variance.lerp_(update, beta_t)
        update = variance.rsqrt()

    update = update.mul_(grad).nan_to_num_().clamp_(-clip,clip)
    update = apply_norm_to_update_(update, param, norm_mode, clips)

    if exp_avg is not None:
        exp_avg, exp_avg_fp32 = lerp_buffer_stochastic_(exp_avg, update, 1 - beta2, use_stochastic_rounding=use_stochastic_buffers)
        update = exp_avg_fp32.clone()
        del exp_avg_fp32

    return update


def approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    return torch.mul(
        torch.div(exp_avg_sq_row, exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1),
        exp_avg_sq_col.rsqrt().unsqueeze(-2),
    )
