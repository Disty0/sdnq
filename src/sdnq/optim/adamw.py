from collections.abc import Iterator

import torch

from ..common import compile_func

from .optimizer import SDNQOptimizer
from .utils import create_quantized_buffer, lerp_buffer_stochastic_


class AdamW(SDNQOptimizer):
    _extra_group_keys = {} # noqa: RUF012
    _keep_in_fp32_keys = {} # noqa: RUF012
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
        super().__init__(param_groups, {})

    @torch.no_grad()
    def init_state(self, param: torch.Tensor, group: dict, state: dict) -> dict:
        use_quantized_buffers = group["use_quantized_buffers"] and param.grad.ndim >= group["quantized_buffers_minimum_ndim"] and param.grad.numel() >= group["quantized_buffers_minimum_numel"]
        if use_quantized_buffers:
            state["exp_avg"] = create_quantized_buffer(torch.zeros_like(param, dtype=torch.float32), group)
            state["exp_avg_sq"] = create_quantized_buffer(torch.zeros_like(param, dtype=torch.float32), group)
        else:
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
        return state

    @torch.no_grad()
    def get_param_update(self, param_fp32: torch.FloatTensor, grad: torch.FloatTensor, group: dict, state: dict) -> torch.FloatTensor:
        update_func = adam_update_compiled if group["use_torch_compile"] else adam_update
        return update_func(
            grad=grad,
            exp_avg=state["exp_avg"],
            exp_avg_sq=state["exp_avg_sq"],
            step=state["step"],
            betas=group["betas"],
            clip=group["clip_threshold"][0],
            use_stochastic_buffers=group["use_stochastic_buffers"],
        )


def adam_update(
    grad: torch.FloatTensor,
    exp_avg: torch.FloatTensor,
    exp_avg_sq: torch.FloatTensor,
    step: int,
    betas: tuple[float, float],
    clip: float,
    use_stochastic_buffers: bool = False,
) -> torch.FloatTensor:
    beta1, beta2 = betas

    exp_avg, exp_avg_fp32 = lerp_buffer_stochastic_(exp_avg, grad, 1 - beta1, use_stochastic_rounding=use_stochastic_buffers)
    exp_avg_c = exp_avg_fp32 / (1 - beta1 ** step)
    del exp_avg_fp32

    exp_avg_sq, exp_avg_sq_fp32 = lerp_buffer_stochastic_(exp_avg_sq, grad.square(), 1 - beta2, use_stochastic_rounding=use_stochastic_buffers)
    exp_avg_sq_c = exp_avg_sq_fp32 / (1 - beta2 ** step)
    del exp_avg_sq_fp32

    return exp_avg_c.mul_(exp_avg_sq_c.rsqrt_()).nan_to_num_().clamp_(-clip,clip)


adam_update_compiled = compile_func(adam_update)
