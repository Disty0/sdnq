from typing import Tuple, Optional

import torch

from .optimizer import SDNQOptimizer


class Adafactor(SDNQOptimizer):
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
            group["clip_threshold"] = group.get("clip_threshold", (1.0, 1e-3, 1e-3))
            group["use_cautious"] = group.get("use_cautious", False)
            group["bf16_stochastic_round"] = group.get("bf16_stochastic_round", False)
            assert set(group.keys()) == set(["params", "lr", "betas", "weight_decay", "clip_threshold", "use_cautious", "bf16_stochastic_round"])
        super().__init__(param_groups, dict())
        self.keep_in_fp32_keys = {"variance", "row_var", "col_var"}

    @torch.no_grad()
    def step(self, closure=None):
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

                state["step"] += 1
                param_fp32 = param.to(dtype=torch.float32)
                grad = param.grad.to(dtype=torch.float32)
                update = adafactor_update(
                    param=param_fp32,
                    grad=grad,
                    row_var=state["row_var"] if factored else None,
                    col_var=state["col_var"] if factored else None,
                    variance=state["variance"] if not factored else None,
                    step=state["step"],
                    betas=group["betas"],
                    clips=group["clip_threshold"][:-1],
                ).to(dtype=torch.float32)


                self.update_param_(
                    param=param,
                    param_fp32=param_fp32,
                    grad=grad,
                    update=update,
                    learning_rate=group["lr"],
                    weight_decay=group["weight_decay"],
                    cautious_clip=group["clip_threshold"][-1],
                    use_cautious=group["use_cautious"],
                    bf16_stochastic_round=group["bf16_stochastic_round"]
                )

        return loss


def adafactor_update(
    param: torch.FloatTensor,
    grad: torch.FloatTensor,
    row_var: torch.FloatTensor,
    col_var: torch.FloatTensor,
    variance: Optional[torch.FloatTensor],
    step: int,
    betas: float,
    clips: Tuple[float, float],
) -> torch.FloatTensor:
    clip, clip2 = clips

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
    update = update.mul_(param.norm(2).clamp_(min=clip2).div_(update.norm(2).clamp_(min=1/clip)))
    return update


def approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    r_factor = torch.div(exp_avg_sq_row, exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
    c_factor = exp_avg_sq_col.rsqrt().unsqueeze(-2)
    return torch.mul(r_factor, c_factor)
