from typing import Tuple

import torch

from .stochastic import copy_stochastic_
from sdnq.training import SDNQTensor


class Lion(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        if isinstance(params, torch.nn.Parameter) or (isinstance(params, list) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            group["lr"] = group.get("lr", 1e-4)
            group["betas"] = group.get("betas", (0.9, 0.95))
            group["weight_decay"] = group.get("weight_decay", 0.01)
            group["bf16_stochastic_round"] = group.get("bf16_stochastic_round", False)
            group["use_quantized_buffers"] = group.get("use_quantized_buffers", False)
            group["quantized_buffers_dtype"] = group.get("quantized_buffers_dtype", "uint8")
            group["quantized_buffers_group_size"] = group.get("quantized_buffers_group_size", 32)
            group["use_stochastic_quantization"] = group.get("use_stochastic_quantization", True)
            assert set(group.keys()) == set(["params", "lr", "betas", "weight_decay", "bf16_stochastic_round", "use_quantized_buffers", "quantized_buffers_dtype", "quantized_buffers_group_size", "use_stochastic_quantization"])
        super().__init__(param_groups, dict())

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            if group["use_quantized_buffers"]:
                for p in group["params"]:
                    state = self.state.get(p, None)
                    if state is not None:
                            state["exp_avg"] = state["exp_avg"].to(dtype=torch.float32)

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
                        state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(p, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], sr=group["use_stochastic_quantization"])
                    else:
                        state["exp_avg"] = torch.zeros_like(p)

                state["step"] += 1
                p_fp32 = p.to(dtype=torch.float32)
                update = lion_update(p.grad, state["exp_avg"], group["betas"]).to(dtype=torch.float32)

                if group["weight_decay"] != 0:
                    p_fp32.mul_(1 - group["lr"] * group["weight_decay"])
                p_fp32.add_(update, alpha=-group["lr"])
                if group["bf16_stochastic_round"]:
                    copy_stochastic_(p, p_fp32)
                else:
                    p.copy_(p_fp32)
                del p_fp32

        return loss


def lion_update(grad: torch.FloatTensor, exp_avg: torch.FloatTensor, betas: Tuple[float, float]) -> torch.FloatTensor:
    beta1, beta2 = betas
    grad = grad.to(dtype=exp_avg.dtype)
    update = exp_avg.lerp(grad, 1 - beta1).sign_()
    exp_avg.lerp_(grad, 1 - beta2)
    return update
