from typing import Tuple, Iterator

import torch

from .optimizer import SDNQOptimizer
from sdnq.training import SDNQTensor


class Lion(SDNQOptimizer):
    _extra_group_keys = {}
    _keep_in_fp32_keys = {}
    _group_keys = set.union(SDNQOptimizer._base_group_keys, _extra_group_keys)

    def __init__(self, params, **kwargs):
        if isinstance(params, (torch.nn.Parameter, Iterator)) or (isinstance(params, list) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            group = self.apply_group_defaults(group)
            assert set(group.keys()) == self._group_keys
        super().__init__(param_groups, dict())
        self.keep_in_fp32_keys = {}

    @torch.no_grad()
    def step(self, closure=None):
        grad_scale = getattr(self, "grad_scale", None)
        found_inf = getattr(self, "found_inf", 0)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None or found_inf > 0:
                    continue

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    if group["use_quantized_buffers"]:
                        state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], sr=group["use_stochastic_quantization"])
                    else:
                        state["exp_avg"] = torch.zeros_like(param)

                state["step"] += 1
                param_fp32 = param.to(dtype=torch.float32)
                grad = param.grad.to(dtype=torch.float32)
                if grad_scale is not None:
                    grad.div_(grad_scale.to(dtype=torch.float32))

                update = lion_update(
                    grad=grad,
                    exp_avg=state["exp_avg"],
                    betas=group["betas"],
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


def lion_update(grad: torch.FloatTensor, exp_avg: torch.FloatTensor, betas: Tuple[float, float]) -> torch.FloatTensor:
    beta1, beta2 = betas
    update = exp_avg.to(dtype=torch.float32).lerp(grad, 1 - beta1).sign_()
    if exp_avg.dtype != torch.float32:
        exp_avg.copy_(exp_avg.to(dtype=torch.float32).lerp_(grad, 1 - beta2))
    else:
        exp_avg.lerp_(grad, 1 - beta2)
    return update
