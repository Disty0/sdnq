from typing import Optional, Tuple, Union

import torch

from ..common import dtype_dict, torch_dtype_dict
from ..training import SDNQTensor


def get_param_grad(
    param: torch.nn.Parameter,
    clip: float = 1.0,
    grad_scale: Optional[torch.FloatTensor] = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    grad = param.grad.nan_to_num_().to(dtype=torch.float32)
    if grad_scale:
        grad.div_(grad_scale.to(dtype=torch.float32))
    grad = grad.clamp_(-clip,clip)

    if isinstance(param, SDNQTensor):
        param_fp32 = param.dequantize(dtype=torch.float32).nan_to_num_()
        if param.dtype not in {torch.float32, torch.bfloat16}:
            max_val = torch.finfo(param.dtype).max
            param_fp32 = param_fp32.clamp_(-max_val, max_val)
    else:
        param_fp32 = param.nan_to_num_().to(dtype=torch.float32)
    return param_fp32, grad


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
    use_stochastic_rounding: bool,
) -> torch.FloatTensor:
    update = apply_norm_to_update_(update, param_fp32, final_norm_mode, clips)
    if use_cautious:
        mask = (torch.mul(update, grad) > 0).to(dtype=torch.float32)
        mask.div_(mask.mean().clamp_(min=clips[-1]))
        update = update.mul_(mask)
    if weight_decay != 0:
        param_fp32.mul_(1 - learning_rate * weight_decay)

    param_fp32.add_(update, alpha=-learning_rate)
    copy_stochastic_(param, param_fp32, use_stochastic_rounding=use_stochastic_rounding)
    return param


def copy_stochastic_(
    target: torch.FloatTensor,
    source: torch.FloatTensor,
    use_stochastic_rounding: bool = True,
) -> torch.FloatTensor:
    if not use_stochastic_rounding or target.dtype == torch.float32 or isinstance(target, SDNQTensor):
        return target.copy_(source)

    target_dtype = torch_dtype_dict[target.dtype]
    min_val = dtype_dict[target_dtype]["min"]
    max_val = dtype_dict[target_dtype]["max"]

    if dtype_dict[target_dtype]["is_integer"]:
        if source.dtype != torch.float32:
            return target.copy_(source.to(dtype=torch.float32).add_(torch.randn_like(source, dtype=torch.float32), alpha=0.1).round_().clamp_(min_val,max_val))
        else:
            return target.copy_(source.add(torch.randn_like(source), alpha=0.1).round_().clamp_(min_val,max_val))
    else:
        mantissa_difference = 1 << (23 - dtype_dict[target_dtype]["mantissa"])
        return target.copy_(
            torch.randint_like(source, low=0, high=mantissa_difference, dtype=torch.int32).add_(source.to(dtype=torch.float32).view(dtype=torch.int32)).view(dtype=torch.float32).clamp_(min_val,max_val)
        )


def lerp_buffer_stochastic_(
    buffer: torch.FloatTensor,
    update: torch.FloatTensor,
    weight: Union[torch.FloatTensor, float],
    use_stochastic_rounding: bool = True,
    return_dequantized_buffer: bool = True,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    if isinstance(buffer, SDNQTensor):
        buffer_fp32 = buffer.dequantize(dtype=torch.float32).lerp_(update, weight)
        buffer.copy_(buffer_fp32)
    elif buffer.dtype != torch.float32:
        buffer_fp32 = buffer.to(dtype=torch.float32).lerp_(update, weight)
        copy_stochastic_(buffer, buffer_fp32, use_stochastic_rounding=use_stochastic_rounding)
    else:
        buffer.lerp_(update, weight)
        buffer_fp32 = buffer
    return buffer, buffer_fp32


def apply_norm_to_update_(update: torch.FloatTensor, param: torch.FloatTensor, norm_mode: str, clips: Tuple[float]) -> torch.FloatTensor:
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
        raise NotImplementedError(f"Norm mode {norm_mode} is not implemented")
    return update.nan_to_num_().clamp_(-clip,clip)


def send_buffers_to_device(state: dict, device: torch.device, non_blocking: bool) -> dict:
    for key, value in state.items():
        if isinstance(value, torch.Tensor) and value.device != device:
            state[key] = value.to(device, non_blocking=non_blocking)
    return state


def send_buffers_to_cpu(state: dict, non_blocking: bool) -> dict:
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            state[key] = value.to("cpu", non_blocking=non_blocking)
    return state
