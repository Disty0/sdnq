from typing import Tuple

import torch

from .stochastic import copy_stochastic_
from sdnq.training import SDNQTensor


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        if isinstance(params, torch.nn.Parameter) or (isinstance(params, list) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            group["lr"] = group.get("lr", 1e-4)
            group["eps"] = group.get("eps", 1e-8)
            group["betas"] = group.get("betas", (0.9, 0.95))
            group["weight_decay"] = group.get("weight_decay", 0)
            group["clip_threshold"] = group.get("clip_threshold", 1)
            group["bf16_stochastic_round"] = group.get("bf16_stochastic_round", False)
            group["use_quantized_buffers"] = group.get("use_quantized_buffers", False)
            group["quantized_buffers_dtype"] = group.get("quantized_buffers_dtype", "int8")
            group["use_stochastic_quantization"] = group.get("use_stochastic_quantization", True)
            assert set(group.keys()) == set(["params", "lr", "eps", "betas", "weight_decay", "clip_threshold", "bf16_stochastic_round", "use_quantized_buffers", "quantized_buffers_dtype", "use_stochastic_quantization"])
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
                if len(state) == 0:
                    state["step"] = 0
                    if group["use_quantized_buffers"]:
                        state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(p), qtype=group["quantized_buffers_dtype"], sr=group["use_stochastic_quantization"])
                        state["exp_avg_sq"] = SDNQTensor.from_float(torch.zeros_like(p), qtype=group["quantized_buffers_dtype"], sr=group["use_stochastic_quantization"])
                    else:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                update = adam_update(
                    p.grad,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    group["betas"],
                    group["eps"],
                    group["clip_threshold"],
                )

                if group["bf16_stochastic_round"]:
                    p_fp32 = p.to(torch.float32)
                    if group["weight_decay"] != 0:
                        p_fp32.mul_(1 - group["lr"] * group["weight_decay"])
                    p_fp32.add_(update, alpha=-group["lr"])
                    copy_stochastic_(p, p_fp32)
                else:
                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


def adam_update(grad: torch.FloatTensor, buf1: torch.FloatTensor, buf2: torch.FloatTensor, step: int, betas: Tuple[float, float], eps: float, clip: float) -> torch.FloatTensor:
    beta, beta2 = betas
    buf1.lerp_(grad, 1 - beta)
    buf2.lerp_(grad.square(), 1 - beta2)
    buf1c = buf1 / (1 - beta ** step)
    buf2c = buf2 / (1 - beta2 ** step)
    return buf1c.mul_(buf2c.rsqrt_().nan_to_num_()).clamp_(-clip,clip)
