from typing import Any, Tuple, Optional

from collections import defaultdict
from collections.abc import Hashable, Iterable
from copy import deepcopy
from itertools import chain

import torch

from .stochastic import copy_stochastic_
from ..training import SDNQTensor


def apply_norm_to_update_(update: torch.FloatTensor, param: torch.FloatTensor, norm_mode: str, clips: Tuple[float]):
    if isinstance(clips, float):
        clip, clip2 = clips, 0
    elif len(clips) == 1:
        clip, clip2 = clips[0], 0
    else:
        clip, clip2 = clips[:2]

    if norm_mode == "none":
        return update.nan_to_num_().clamp_(-clip,clip)
    elif norm_mode == "rms":
        update = update.mul_(torch.div((clip * update.numel()**0.5), update.norm(2)))
    elif norm_mode == "rms_clip":
        update = update.mul_(torch.div((clip * update.numel()**0.5), update.norm(2)).clamp_(max=1))
    elif norm_mode in {"relative", "adafactor"}:
        update = update.mul_(param.norm(2).clamp_(min=clip2).div_(update.norm(2).clamp_(min=1/clip)))
    elif norm_mode in {"rms_scaled", "adamuon"}:
        return apply_norm_to_update_(update, param, "rms", clip * 0.2)
    elif norm_mode in {"rms_clip_scaled", "adamuon_clip"}:
        return apply_norm_to_update_(update, param, "rms_clip", clip * 0.2)
    elif norm_mode == "muon":
        output_shape = update.shape[0]
        input_shape = 1
        for shape in update.shape[1:]:
            input_shape *= shape
        update = update.mul_(max(1, output_shape / input_shape)**0.5)
    else:
        raise NotImplementedError(f'Norm mode {norm_mode} is not implemented')
    return update.nan_to_num_().clamp_(-clip,clip)


class SDNQOptimizer(torch.optim.Optimizer):
    _base_group_keys = {"params", "lr", "betas", "weight_decay", "clip_threshold", "final_norm_mode", "use_cautious", "use_stochastic_rounding", "use_stochastic_buffers", "use_quantized_buffers", "quantized_buffers_dtype", "quantized_buffers_group_size", "quantized_buffers_svd_rank", "use_svd_quantization"}
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
        group["use_cautious"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "use_cautious", False)
        group["use_stochastic_rounding"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "use_stochastic_rounding", True)
        group["use_stochastic_buffers"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "use_stochastic_buffers", True)
        group["use_quantized_buffers"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "use_quantized_buffers", False)
        group["quantized_buffers_dtype"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "quantized_buffers_dtype", "uint8")
        group["quantized_buffers_group_size"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "quantized_buffers_group_size", 32)
        group["quantized_buffers_svd_rank"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "quantized_buffers_svd_rank", 32)
        group["use_svd_quantization"] = SDNQOptimizer.get_default_kwarg(group, kwargs, "use_svd_quantization", False)
        return group

    @staticmethod
    def get_param_grad(
        param: torch.nn.Parameter,
        clip: float = 1.0,
        grad_scale: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        grad = param.grad.nan_to_num_().to(dtype=torch.float32)
        if grad_scale is not None:
            grad.div_(grad_scale.to(dtype=torch.float32))
        grad = grad.clamp_(-clip,clip)

        if isinstance(param, SDNQTensor):
            param_fp32 = param.dequantize(dtype=torch.float32)
        else:
            param_fp32 = param.to(dtype=torch.float32)
        return param_fp32, grad

    @staticmethod
    def update_param_(
        param: torch.nn.Parameter,
        param_fp32: torch.FloatTensor,
        grad: torch.FloatTensor,
        update: torch.FloatTensor,
        learning_rate: float,
        weight_decay: float,
        clips: Tuple[float],
        final_norm_mode: str,
        use_cautious: bool,
        use_stochastic_rounding: bool
    ) -> None:
        update = apply_norm_to_update_(update, param_fp32, final_norm_mode, clips)
        if use_cautious:
            mask = (torch.mul(update, grad) > 0).to(dtype=torch.float32)
            mask.div_(mask.mean().clamp_(min=clips[-1]))
            update = update.mul_(mask)
        if weight_decay != 0:
            param_fp32.mul_(1 - learning_rate * weight_decay)

        param_fp32.add_(update, alpha=-learning_rate)
        if use_stochastic_rounding and param.dtype in {torch.float16, torch.bfloat16} and not isinstance(param, SDNQTensor):
            copy_stochastic_(param, param_fp32)
        else:
            param.copy_(param_fp32)

    def _process_value_according_to_param_policy(self, param: torch.Tensor, value: torch.Tensor, param_id: int, param_groups: list[dict[Any, Any]], key: Hashable = None) -> torch.Tensor:
        assert param_groups is not None
        if key == "step":
            return value
        elif isinstance(value, SDNQTensor) or key in self._keep_in_fp32_keys:
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
