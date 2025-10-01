from typing import Tuple, Optional

import torch

from .optimizer import SDNQOptimizer, apply_norm_to_update_
from sdnq.training import SDNQTensor


class Adafactor(SDNQOptimizer):
    def __init__(self, params, **kwargs):
        if isinstance(params, torch.nn.Parameter) or (isinstance(params, list) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            group["lr"] = group.get("lr", 1e-2)
            group["betas"] = group.get("betas", (-0.8, 0.95))
            group["weight_decay"] = group.get("weight_decay", 0.01)
            group["clip_threshold"] = group.get("clip_threshold", (1.0, 1e-3, 1e-3))
            group["norm_mode"] = group.get("norm_mode", "relative")
            group["final_norm_mode"] = group.get("final_norm_mode", "none")
            group["use_first_moment"] = group.get("use_first_moment", False)
            group["use_cautious"] = group.get("use_cautious", False)
            group["bf16_stochastic_round"] = group.get("bf16_stochastic_round", False)
            group["use_quantized_buffers"] = group.get("use_quantized_buffers", False)
            group["quantized_buffers_dtype"] = group.get("quantized_buffers_dtype", "uint8")
            group["quantized_buffers_group_size"] = group.get("quantized_buffers_group_size", 32)
            group["use_stochastic_quantization"] = group.get("use_stochastic_quantization", True)
            assert set(group.keys()) == set(["params", "lr", "betas", "weight_decay", "clip_threshold", "norm_mode", "final_norm_mode", "use_first_moment", "use_cautious", "bf16_stochastic_round", "use_quantized_buffers", "quantized_buffers_dtype", "quantized_buffers_group_size", "use_stochastic_quantization"])
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

                    if group["use_first_moment"]:
                        if group["use_quantized_buffers"]:
                            state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], sr=group["use_stochastic_quantization"])
                        else:
                            state["exp_avg"] = torch.zeros_like(param)

                state["step"] += 1
                param_fp32 = param.to(dtype=torch.float32)
                grad = param.grad.to(dtype=torch.float32)
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
                    bf16_stochastic_round=group["bf16_stochastic_round"],
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
        if isinstance(exp_avg, SDNQTensor):
            exp_avg_fp32 = exp_avg.dequantize(dtype=torch.float32).lerp_(update, 1 - beta2)
            exp_avg.copy_(exp_avg_fp32)
            update = exp_avg_fp32
        elif exp_avg.dtype == torch.float32:
            exp_avg = exp_avg.lerp_(update, 1 - beta2)
            update = exp_avg.clone()
        else:
            exp_avg_fp32 = exp_avg.to(dtype=torch.float32).lerp_(update, 1 - beta2)
            exp_avg.copy_(exp_avg_fp32)
            update = exp_avg_fp32

    return update


def approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
    r_factor = torch.div(exp_avg_sq_row, exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
    c_factor = exp_avg_sq_col.rsqrt().unsqueeze(-2)
    return torch.mul(r_factor, c_factor)
