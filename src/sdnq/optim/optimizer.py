from typing import Any

from collections import defaultdict
from collections.abc import Hashable, Iterable
from copy import deepcopy
from itertools import chain

import torch

from ..training import SDNQTensor
from .utils import get_param_grad, update_param_, send_buffers_to_device, send_buffers_to_cpu


class SDNQOptimizer(torch.optim.Optimizer):
    _base_group_keys = {"params", "lr", "betas", "weight_decay", "clip_threshold", "final_norm_mode", "use_kahan", "use_cautious", "use_stochastic_rounding", "use_stochastic_buffers", "use_quantized_buffers", "quantized_buffers_dtype", "quantized_buffers_group_size", "quantized_buffers_svd_rank", "use_svd_quantization", "offload_buffers", "offload_non_blocking", "offload_non_blocking_cpu"}
    _extra_group_keys = {}
    _keep_in_fp32_keys = {}
    _group_keys = set.union(_base_group_keys, _extra_group_keys)
    _step_supports_amp_scaling = True

    @staticmethod
    def get_default_kwarg(group: dict, kwargs: dict, key: str, default):
        return group.get(key, kwargs.get(key, default))

    @staticmethod
    def apply_group_defaults(group: dict, **kwargs) -> dict:
        group["lr"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "lr", 1e-4)
        group["betas"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "betas", (0.9, 0.95))
        group["weight_decay"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "weight_decay", 0.01)
        group["clip_threshold"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "clip_threshold", (1.0, 1e-3, 1e-3))
        group["final_norm_mode"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "final_norm_mode", "none")
        group["use_kahan"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "use_kahan", False)
        group["use_cautious"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "use_cautious", False)
        group["use_stochastic_rounding"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "use_stochastic_rounding", True)
        group["use_stochastic_buffers"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "use_stochastic_buffers", True)
        group["use_quantized_buffers"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "use_quantized_buffers", False)
        group["quantized_buffers_dtype"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "quantized_buffers_dtype", "uint8")
        group["quantized_buffers_group_size"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "quantized_buffers_group_size", 32)
        group["quantized_buffers_svd_rank"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "quantized_buffers_svd_rank", 32)
        group["use_svd_quantization"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "use_svd_quantization", False)
        group["offload_buffers"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "offload_buffers", False)
        group["offload_non_blocking"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "offload_non_blocking", True)
        group["offload_non_blocking_cpu"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "offload_non_blocking_cpu", group["offload_non_blocking"])
        return group

    @torch.no_grad()
    def init_state(self, param: torch.Tensor, group: dict, state: dict) -> dict:
        raise NotImplementedError
        return state

    @torch.no_grad()
    def get_param_update(self, param_fp32: torch.FloatTensor, grad: torch.FloatTensor, group: dict, state: dict) -> torch.FloatTensor:
        raise NotImplementedError
        return update # noqa: F821

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
                    state = self.init_state(param, group, state)
                    if group["use_kahan"]:
                        if group["use_quantized_buffers"]:
                            state["kahan_buffer"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), weights_dtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], use_stochastic_rounding=group["use_stochastic_buffers"])
                        else:
                            state["kahan_buffer"] = torch.zeros_like(param)

                state["step"] += 1
                state = send_buffers_to_device(state, param.device, group["offload_non_blocking"])
                param_fp32, grad = get_param_grad(param, clip=group["clip_threshold"][0], grad_scale=grad_scale)
                update = self.get_param_update(param_fp32, grad, group, state).to(dtype=torch.float32)

                if group["offload_buffers"]:
                    state = send_buffers_to_cpu(state, group["offload_non_blocking_cpu"])
                    if group["use_kahan"] and state["kahan_buffer"].device != param.device:
                        state["kahan_buffer"] = state["kahan_buffer"].to(param.device, non_blocking=group["offload_non_blocking"])

                update_param_(
                    param=param,
                    param_fp32=param_fp32,
                    grad=grad,
                    update=update,
                    kahan_buffer=state.get("kahan_buffer", None),
                    learning_rate=group["lr"],
                    weight_decay=group["weight_decay"],
                    clips=group["clip_threshold"],
                    final_norm_mode=group["final_norm_mode"],
                    use_cautious=group["use_cautious"],
                    use_stochastic_rounding=group["use_stochastic_rounding"],
                    use_stochastic_buffers=group["use_stochastic_buffers"],
                )

                if group["offload_buffers"] and group["use_kahan"] and state["kahan_buffer"].device.type != "cpu":
                    state["kahan_buffer"] = state["kahan_buffer"].to("cpu", non_blocking=group["offload_non_blocking_cpu"])

        return loss

    def _process_value_according_to_param_policy(self, param: torch.Tensor, value: torch.Tensor, param_id: int, param_groups: list[dict[Any, Any]], key: Hashable = None, device: torch.device = None) -> torch.Tensor:
        if device is None:
            device = param.device
        if key == "step":
            return value
        elif param.dtype == torch.float32 or isinstance(value, SDNQTensor) or key in self._keep_in_fp32_keys:
            # Sending in 16 bit to GPU and casting to FP32 in GPU is much faster than sending it directly in FP32
            return value.to(device=device).to(dtype=torch.float32)
        else:
            return value.to(dtype=param.dtype).to(device=device)

    def _load_state_dict_cast(self, param, value, param_id=None, param_groups=None, key=None, device=None):
        r"""Make a deep copy of value, casting all tensors to device of param."""
        if isinstance(value, torch.Tensor):
            return self._process_value_according_to_param_policy(param, value, param_id, param_groups, key=key, device=device)
        elif isinstance(value, dict):
            return {k: self._load_state_dict_cast(param, v, param_id=param_id, param_groups=param_groups, key=k, device=device) for k, v in value.items()}
        elif isinstance(value, Iterable):
            return type(value)(self._load_state_dict_cast(param, v, param_id=param_id, param_groups=param_groups, device=device) for v in value) # type: ignore[call-arg]
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
        device = "cpu" if any(group["offload_buffers"] for group in state_dict["param_groups"]) else None

        state: defaultdict[torch.Tensor, dict[Any, Any]] = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = self._load_state_dict_cast(param, v, param_id=k, param_groups=state_dict["param_groups"], device=device)
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
