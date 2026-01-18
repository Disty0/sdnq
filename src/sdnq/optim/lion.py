from typing import Tuple, Iterator

import torch

from ..training import SDNQTensor
from ..common import compile_func

from .optimizer import SDNQOptimizer
from .utils import lerp_buffer_stochastic_


class Lion(SDNQOptimizer):
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
        self.keep_in_fp32_keys = {}

    @torch.no_grad()
    def init_state(self, param: torch.Tensor, group: dict, state: dict) -> dict:
        if group["use_quantized_buffers"]:
            state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), weights_dtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], use_stochastic_rounding=group["use_stochastic_buffers"])
        else:
            state["exp_avg"] = torch.zeros_like(param)
        return state

    @torch.no_grad()
    def get_param_update(self, param_fp32: torch.FloatTensor, grad: torch.FloatTensor, group: dict, state: dict) -> torch.FloatTensor:
        update_func = lion_update_compiled if group["use_torch_compile"] else lion_update
        return update_func(
            grad=grad,
            exp_avg=state["exp_avg"],
            betas=group["betas"],
            use_stochastic_buffers=group["use_stochastic_buffers"],
        )


def lion_update(
    grad: torch.FloatTensor,
    exp_avg: torch.FloatTensor,
    betas: Tuple[float, float],
    use_stochastic_buffers: bool = False,
) -> torch.FloatTensor:
    beta1, beta2 = betas
    update = exp_avg.to(dtype=torch.float32).lerp(grad, 1 - beta1).sign_()
    lerp_buffer_stochastic_(exp_avg, grad, 1 - beta2, use_stochastic_rounding=use_stochastic_buffers)
    return update


lion_update_compiled = compile_func(lion_update)
