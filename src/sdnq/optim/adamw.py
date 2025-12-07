from typing import Tuple, Iterator

import torch

from .optimizer import SDNQOptimizer
from .stochastic import copy_stochastic_
from ..training import SDNQTensor


class AdamW(SDNQOptimizer):
    _extra_group_keys = {}
    _keep_in_fp32_keys = {}
    _group_keys = set.union(SDNQOptimizer._base_group_keys, _extra_group_keys)

    def __init__(self, params, **kwargs):
        if isinstance(params, (torch.nn.Parameter, Iterator)) or (isinstance(params, (list, tuple)) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
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
                if len(state) == 0:
                    state["step"] = 0
                    if group["use_quantized_buffers"]:
                        state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), weights_dtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], use_stochastic_rounding=group["use_stochastic_buffers"])
                        state["exp_avg_sq"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), weights_dtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], use_stochastic_rounding=group["use_stochastic_buffers"])
                    else:
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)

                state["step"] += 1
                param_fp32, grad = self.get_param_grad(param, clip=group["clip_threshold"][0], grad_scale=grad_scale)

                update = adam_update(
                    grad=grad,
                    exp_avg=state["exp_avg"],
                    exp_avg_sq=state["exp_avg_sq"],
                    step=state["step"],
                    betas=group["betas"],
                    clip=group["clip_threshold"][0],
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


def adam_update(
    grad: torch.FloatTensor,
    exp_avg: torch.FloatTensor,
    exp_avg_sq: torch.FloatTensor,
    step: int,
    betas: Tuple[float, float],
    clip: float,
    use_stochastic_buffers: bool = False,
) -> torch.FloatTensor:
    beta1, beta2 = betas
    if exp_avg.dtype != torch.float32:
        exp_avg_fp32 = exp_avg.to(dtype=torch.float32).lerp_(grad, 1 - beta1)
        exp_avg_sq_fp32 = exp_avg_sq.to(dtype=torch.float32).lerp_(grad.square(), 1 - beta2)
        if use_stochastic_buffers and not isinstance(exp_avg, SDNQTensor):
            copy_stochastic_(exp_avg, exp_avg_fp32)
            copy_stochastic_(exp_avg_sq, exp_avg_sq_fp32)
        else:
            exp_avg.copy_(exp_avg_fp32)
            exp_avg_sq.copy_(exp_avg_sq_fp32)
    else:
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.lerp_(grad.square(), 1 - beta2)
        exp_avg_fp32 = exp_avg
        exp_avg_sq_fp32 = exp_avg_sq
    exp_avg_c = exp_avg_fp32 / (1 - beta1 ** step)
    exp_avg_sq_c = exp_avg_sq_fp32 / (1 - beta2 ** step)
    return exp_avg_c.mul_(exp_avg_sq_c.rsqrt_()).nan_to_num_().clamp_(-clip,clip)
