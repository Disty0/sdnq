from collections.abc import Iterator

import torch

from ..sdnext import devices
from ..common import compile_func

from .optimizer import SDNQOptimizer
from .utils import create_quantized_buffer, lerp_buffer_stochastic_


class Lion(SDNQOptimizer):
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
        self.keep_in_fp32_keys = {}

    @devices.inference_context()
    def init_state(self, param: torch.Tensor, group: dict, state: dict) -> dict:
        use_quantized_buffers = group["use_quantized_buffers"] and param.grad.ndim >= group["quantized_buffers_minimum_ndim"] and param.grad.numel() >= group["quantized_buffers_minimum_numel"]
        if use_quantized_buffers:
            state["exp_avg"] = create_quantized_buffer(torch.zeros_like(param, dtype=torch.float32), group)
        else:
            state["exp_avg"] = torch.zeros_like(param)
        return state

    @devices.inference_context()
    def get_param_update(self, param_fp32: torch.FloatTensor, grad: torch.FloatTensor, group: dict, state: dict) -> torch.FloatTensor:
        update_func = lion_update_compiled if group["use_torch_compile"] else lion_update
        return update_func(
            grad=grad,
            exp_avg=state["exp_avg"],
            betas=group["betas"],
            use_stochastic_buffers=group["use_stochastic_buffers"],
        )


@devices.inference_context()
def lion_update(
    grad: torch.FloatTensor,
    exp_avg: torch.FloatTensor,
    betas: tuple[float, float],
    use_stochastic_buffers: bool = False,
) -> torch.FloatTensor:
    beta1, beta2 = betas
    update = exp_avg.to(dtype=torch.float32).lerp(grad, 1 - beta1).sign_()
    lerp_buffer_stochastic_(exp_avg, grad, 1 - beta2, use_stochastic_rounding=use_stochastic_buffers)
    return update


lion_update_compiled = compile_func(lion_update)
