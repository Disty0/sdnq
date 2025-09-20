import torch

from .stochastic import copy_stochastic_


class AdafactorBF16(torch.optim.Optimizer):
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

                factored = p.ndim >= 2
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    if factored:
                        row_shape = list(p.shape)
                        col_shape = list(p.shape)
                        row_shape[-1] = 1
                        col_shape[-2] = 1
                        state["row_var"] = torch.zeros(row_shape, dtype=torch.float32, device=p.device)
                        state["col_var"] = torch.zeros(col_shape, dtype=torch.float32, device=p.device)
                    else:
                        state["variance"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                update = adafactor_update(
                    p.to(dtype=torch.float32),
                    p.grad.to(dtype=torch.float32),
                    state["row_var"] if factored else None,
                    state["col_var"] if factored else None,
                    state["variance"] if not factored else None,
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
    param: torch.Tensor,
    grad: torch.Tensor,
    row_var: torch.Tensor,
    col_var: torch.Tensor,
    variance: torch.Tensor,
    step: int,
    betas: float,
    clip: float,
):
    beta_t = step**betas

    if variance is None:
        row_mean = torch.norm(grad, dim=-1, keepdim=True).square_().div_(grad.size(-1))
        row_var.lerp_(row_mean, beta_t)
        col_mean = torch.norm(grad, dim=-2, keepdim=True).square_().div_(grad.size(-2))
        col_var.lerp_(col_mean, beta_t)
        var_estimate = (row_var @ col_var).div_(row_var.mean(dim=-2, keepdim=True)).nan_to_num_()
    else:
        variance.lerp_(grad.square(), beta_t)
        var_estimate = variance.clone()

    update = var_estimate.rsqrt_().mul_(grad).nan_to_num_().clamp_(-clip,clip)
    update = update.mul_(param.norm(2).mul_(clip * 0.2).div_(update.norm(2)))
    return update