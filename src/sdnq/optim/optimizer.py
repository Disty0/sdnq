from typing import Any

from collections import defaultdict
from collections.abc import Hashable, Iterable
from copy import deepcopy
from itertools import chain

import torch

from .stochastic import copy_stochastic_
from sdnq.training import SDNQTensor

# torch.optim.Optimizer calls Optimizer._process_value_according_to_param_policy instead of self._process_value_according_to_param_policy
# so we have to create a whloe new class to modify _process_value_according_to_param_policy
class SDNQOptimizer(torch.optim.Optimizer):

    def _process_value_according_to_param_policy(self, param: torch.Tensor, value: torch.Tensor, param_id: int, param_groups: list[dict[Any, Any]], key: Hashable = None) -> torch.Tensor:
        assert param_groups is not None
        if key == "step":
            return value
        elif isinstance(value, SDNQTensor) or key in self.keep_in_fp32_keys:
            return value.to(device=param.device, dtype=torch.float32)
        else:
            return value.to(device=param.device, dtype=param.dtype)

    def _load_state_dict_cast(self, param, value, param_id=None, param_groups=None, key=None):
        r"""Make a deep copy of value, casting all tensors to device of param."""
        if isinstance(value, torch.Tensor):
            return self._process_value_according_to_param_policy(param, value, param_id, param_groups, key)
        elif isinstance(value, dict):
            return {k: self._load_state_dict_cast(param, v, param_id=param_id, param_groups=param_groups, key=k) for k, v in value.items()}
        elif isinstance(value, Iterable):
            return type(value)(self._load_state_dict_cast(param, v, param_id=param_id, param_groups=param_groups) for v in value) # type: ignore[call-arg]
        else:
            return value

    @torch._disable_dynamo
    def load_state_dict(self, state_dict: dict) -> None:
        # shallow copy, to be consistent with module API
        state_dict = state_dict.copy()

        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result

        # Validate the state_dict
        groups = self.param_groups

        # Deepcopy as we write into saved_groups later to update state
        saved_groups = deepcopy(state_dict["param_groups"])

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of parameter groups")
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group that doesn't match the size of optimizer's group")

        # Update the state
        id_map = dict(zip(chain.from_iterable(g["params"] for g in saved_groups), chain.from_iterable(g["params"] for g in groups)))

        state: defaultdict[torch.Tensor, dict[Any, Any]] = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = self._load_state_dict_cast(param, v, param_id=k, param_groups=state_dict["param_groups"])
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group: dict[str, Any], new_group: dict[str, Any]) -> dict[str, Any]:
            new_group["params"] = group["params"]
            if "param_names" in group and "param_names" not in new_group:
                new_group["param_names"] = group["param_names"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

    @staticmethod
    def update_param_(
        param: torch.nn.Parameter,
        param_fp32: torch.FloatTensor,
        grad: torch.FloatTensor,
        update: torch.FloatTensor,
        learning_rate: float,
        weight_decay: float,
        clip_threshold: float,
        use_cautious: bool,
        bf16_stochastic_round: bool
    ):
        if use_cautious:
            mask = (torch.mul(update, grad) > 0).to(dtype=torch.float32)
            mask.div_(mask.mean().clamp_(min=clip_threshold))
            update.mul_(mask)
        if weight_decay != 0:
            param_fp32.mul_(1 - learning_rate * weight_decay)

        param_fp32.add_(update, alpha=-learning_rate)
        if bf16_stochastic_round:
            copy_stochastic_(param, param_fp32)
        else:
            param.copy_(param_fp32)
