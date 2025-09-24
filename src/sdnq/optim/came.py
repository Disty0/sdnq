from typing import Tuple, Optional

import torch

from .stochastic import copy_stochastic_
from sdnq.training import SDNQTensor


class CAME(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        if isinstance(params, torch.nn.Parameter) or (isinstance(params, list) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            group["lr"] = group.get("lr", 1e-4)
            group["betas"] = group.get("betas", (0.9, 0.999, 0.9999))
            group["weight_decay"] = group.get("weight_decay", 0.01)
            group["clip_threshold"] = group.get("clip_threshold", 1.0)
            group["bf16_stochastic_round"] = group.get("bf16_stochastic_round", False)
            group["use_quantized_buffers"] = group.get("use_quantized_buffers", False)
            group["quantized_buffers_dtype"] = group.get("quantized_buffers_dtype", "uint8")
            group["quantized_buffers_group_size"] = group.get("quantized_buffers_group_size", 32)
            group["use_stochastic_quantization"] = group.get("use_stochastic_quantization", True)
            assert set(group.keys()) == set(["params", "lr", "betas", "weight_decay", "clip_threshold", "bf16_stochastic_round", "use_quantized_buffers", "quantized_buffers_dtype", "quantized_buffers_group_size", "use_stochastic_quantization"])
        super().__init__(param_groups, dict())


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state.get(p, None)
                if state is not None:
                    if group["use_quantized_buffers"]:
                        state["exp_avg"] = state["exp_avg"].to(dtype=torch.float32)
                    if state.get("exp_avg_sq", None) is None:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(dtype=torch.float32)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(dtype=torch.float32)
                        state["exp_avg_res_row"] = state["exp_avg_res_row"].to(dtype=torch.float32)
                        state["exp_avg_res_col"] = state["exp_avg_res_col"].to(dtype=torch.float32)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(dtype=torch.float32)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
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
                    if group["use_quantized_buffers"]:
                        state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(p, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], sr=group["use_stochastic_quantization"])
                    else:
                        state["exp_avg"] = torch.zeros_like(p)

                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1], dtype=torch.float32, device=p.device)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=torch.float32, device=p.device)
                        state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1], dtype=torch.float32, device=p.device)
                        state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=torch.float32, device=p.device)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                p_fp32 = p.to(dtype=torch.float32)
                update = came_update(
                    p.grad,
                    state["exp_avg_sq_row"] if factored else None,
                    state["exp_avg_sq_col"] if factored else None,
                    state["exp_avg_res_row"] if factored else None,
                    state["exp_avg_res_col"] if factored else None,
                    state["exp_avg_sq"] if not factored else None,
                    state["exp_avg"],
                    state["step"],
                    group["betas"],
                    group["clip_threshold"],
                ).to(dtype=torch.float32)

                if group["weight_decay"] != 0:
                    p_fp32.mul_(1 - group["lr"] * group["weight_decay"])
                p_fp32.add_(update, alpha=-group["lr"])
                if group["bf16_stochastic_round"]:
                    copy_stochastic_(p, p_fp32)
                else:
                    p.copy_(p_fp32)

        return loss


def came_update(
    grad: torch.FloatTensor,
    exp_avg_sq_row: torch.FloatTensor,
    exp_avg_sq_col: torch.FloatTensor,
    exp_avg_res_row: torch.FloatTensor,
    exp_avg_res_col: torch.FloatTensor,
    exp_avg_sq: Optional[torch.FloatTensor],
    exp_avg: torch.FloatTensor,
    step: int,
    betas: Tuple[float, float, float],
    clip: float,
) -> torch.FloatTensor:
    beta0, beta1, beta2 = betas
    grad = grad.to(dtype=torch.float32)

    one_minus_beta1 = 1 - beta1
    update = torch.square(grad)
    if exp_avg_sq is None:
        exp_avg_sq_row.lerp_(update.mean(dim=-1), one_minus_beta1)
        exp_avg_sq_col.lerp_(update.mean(dim=-2), one_minus_beta1)
        update = approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
    else:
        exp_avg_sq.lerp_(update, one_minus_beta1)
        update = exp_avg_sq.rsqrt()

    update = update.mul_(grad).nan_to_num_().clamp_(-clip,clip)
    update = update.mul_(torch.div((clip * update.numel()**0.5), update.norm(2)).clamp_(max=1))

    exp_avg.lerp_(update.to(dtype=exp_avg.dtype), 1 - beta0)
    exp_avg_fp32 = exp_avg.to(dtype=torch.float32)
    if exp_avg_sq is None:
        res = torch.sub(update, exp_avg_fp32).square_()
        one_minus_beta2 = 1 - beta2
        exp_avg_res_row.lerp_(res.mean(dim=-1), one_minus_beta2)
        exp_avg_res_col.lerp_(res.mean(dim=-2), one_minus_beta2)
        update = approx_sq_grad(exp_avg_res_row, exp_avg_res_col).mul_(exp_avg_fp32)
    else:
        update = exp_avg_fp32.clone()

    update = update.nan_to_num_().clamp_(-clip,clip)
    return update


def approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    r_factor = torch.div(exp_avg_sq_row, exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
    c_factor = exp_avg_sq_col.rsqrt().unsqueeze(-2)
    return torch.mul(r_factor, c_factor)
