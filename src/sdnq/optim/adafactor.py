from typing import Optional

import torch

from .stochastic import copy_stochastic_


class Adafactor(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        if isinstance(params, torch.nn.Parameter) or (isinstance(params, list) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            group["lr"] = group.get("lr", 1e-2)
            group["betas"] = group.get("betas", -0.8)
            group["weight_decay"] = group.get("weight_decay", 0.01)
            group["clip_threshold"] = group.get("clip_threshold", 1.0)
            group["bf16_stochastic_round"] = group.get("bf16_stochastic_round", False)
            assert set(group.keys()) == set(["params", "lr", "betas", "weight_decay", "clip_threshold", "bf16_stochastic_round"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                grad_shape = p.grad.shape
                factored = len(grad_shape) >= 2

                if len(state) == 0:
                    state["step"] = 0
                    if factored:
                        state["row_var"] = torch.zeros(grad_shape[:-1], dtype=torch.float32, device=p.device)
                        state["col_var"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=torch.float32, device=p.device)
                    else:
                        state["variance"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                update = adafactor_update(
                    p.to(dtype=torch.float32),
                    p.grad.to(dtype=torch.float32),
                    state["row_var"] if factored else None,
                    state["col_var"] if factored else None,
                    state["variance"] if not factored else None,
                    group["lr"],
                    state["step"],
                    group["betas"],
                    group["clip_threshold"],
                )

                if group["bf16_stochastic_round"]:
                    p_fp32 = p.to(dtype=torch.float32)
                    if group["weight_decay"] != 0:
                        p_fp32.mul_(1 - group["lr"] * group["weight_decay"])
                    p_fp32.add_(update, alpha=-min(group["lr"], 1 / (state["step"]**0.5)))
                    copy_stochastic_(p, p_fp32)
                else:
                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-min(group["lr"], 1 / (state["step"]**0.5)))

        return loss


def adafactor_update(
    param: torch.FloatTensor,
    grad: torch.FloatTensor,
    row_var: torch.FloatTensor,
    col_var: torch.FloatTensor,
    variance: Optional[torch.FloatTensor],
    lr: float,
    step: int,
    betas: float,
    clip: float,
):
    beta_t = step**betas
    update = torch.square(grad)
    if variance is None:
        row_var.lerp_(update.mean(dim=-1), beta_t)
        col_var.lerp_(update.mean(dim=-2), beta_t)
        update = approx_sq_grad(row_var, col_var)
    else:
        variance.lerp_(update, beta_t)
        update = variance.rsqrt()

    update = update.mul_(grad).nan_to_num_().clamp_(-clip,clip)
    update = update.mul_(param.norm(2).clamp_(min=lr).div_(update.norm(2).clamp_(min=1/clip)))
    return update


def approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    r_factor = torch.div(exp_avg_sq_row, exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
    c_factor = exp_avg_sq_col.rsqrt().unsqueeze(-2)
    return torch.mul(r_factor, c_factor)
