from typing import Callable, Tuple, Optional, Iterator

import torch

from ..training import SDNQTensor

from ..common import compile_func, use_tensorwise_fp8_matmul
from ..training.layers.linear.linear_int8_dynamic import int8_matmul_dynamic
from ..training.layers.linear.linear_fp8_dynamic import fp8_matmul_dynamic
from ..training.layers.linear.linear_fp8_tensorwise_dynamic import fp8_matmul_tensorwise_dynamic
from ..training.layers.linear.linear_fp16_dynamic import fp16_matmul_dynamic

from .optimizer import SDNQOptimizer
from .utils import get_param_grad, update_param_, lerp_buffer_stochastic_
from .adamw import adam_update


class Muon(SDNQOptimizer):
    _extra_group_keys = [
        {"use_muon", "ns_steps", "nesterov", "adaptive", "zeropower_dtype", "use_quantized_matmul", "quantized_matmul_dtype"},
        {"use_muon"},
    ]
    _keep_in_fp32_keys = {}
    _group_keys = [
        set.union(SDNQOptimizer._base_group_keys, _extra_group_keys[0]),
        set.union(SDNQOptimizer._base_group_keys, _extra_group_keys[1]),
    ]

    def __init__(self, params, **kwargs):
        extra_kwargs = kwargs
        if isinstance(params, (torch.nn.Parameter, Iterator)) or (isinstance(params, (list, tuple)) and isinstance(params[0], torch.nn.Parameter)):
            param_groups, extra_kwargs = self.get_muon_groups(params, **kwargs)
        else:
            param_groups = params

        new_groups = []
        for group in param_groups:
            if "use_muon" not in group:
                extra_kwargs = kwargs.copy()
                for key, value in group.items():
                    extra_kwargs[key] = value
                new_params = extra_kwargs.pop("params", None)
                new_param_group, extra_kwargs = self.get_muon_groups(new_params, **extra_kwargs)
                new_groups.extend(new_param_group)
            else:
                new_groups.append(group)
        param_groups, kwargs = new_groups, extra_kwargs

        for group in param_groups:
            if group["use_muon"]:
                group["lr"] = self.get_default_kwarg(group, kwargs, "lr", 1e-3)
                group["ns_steps"] = self.get_default_kwarg(group, kwargs, "ns_steps", 5)
                group["nesterov"] = self.get_default_kwarg(group, kwargs, "nesterov", True)
                group["adaptive"] = self.get_default_kwarg(group, kwargs, "adaptive", False)
                group["final_norm_mode"] = self.get_default_kwarg(group, kwargs, "final_norm_mode", "rms_clip_scaled")
                group["zeropower_dtype"] = self.get_default_kwarg(group, kwargs, "zeropower_dtype", "bfloat16")
                group["use_quantized_matmul"] = self.get_default_kwarg(group, kwargs, "use_quantized_matmul", False)
                group["quantized_matmul_dtype"] = self.get_default_kwarg(group, kwargs, "quantized_matmul_dtype", "int8")
                if isinstance(group["zeropower_dtype"], str):
                    group["zeropower_dtype"] = getattr(torch, group["zeropower_dtype"])
                group = self.apply_group_defaults(group, **kwargs)
                assert set(group.keys()) == self._group_keys[0]
            else:
                group = self.apply_group_defaults(group, **kwargs)
                assert set(group.keys()) == self._group_keys[1]
        super().__init__(param_groups, dict())

    @staticmethod
    def get_muon_groups(params, **kwargs):
        muon_group = {"use_muon": True, "params": []}
        adamw_group = {"use_muon": False, "params": []}
        extra_kwargs = kwargs.copy()
        keys_to_pop = []
        for key, value in extra_kwargs.items():
            if key.startswith("muon_"):
                muon_group[key.removeprefix("muon_")] = value
                keys_to_pop.append(key)
            elif key.startswith("adamw_"):
                adamw_group[key.removeprefix("adamw_")] = value
                keys_to_pop.append(key)
        for key in keys_to_pop:
            extra_kwargs.pop(key, None)
        for key, value in extra_kwargs.items():
            if key not in muon_group.keys():
                muon_group[key] = value
            if key not in adamw_group.keys() and key not in Muon._extra_group_keys[0]:
                adamw_group[key] = value
        for param in params:
            if param.ndim <= 1:
                adamw_group["params"].append(param)
            else:
                muon_group["params"].append(param)
        return [muon_group, adamw_group], extra_kwargs

    @torch.no_grad()
    def step(self, closure=None):
        grad_scale = getattr(self, "grad_scale", None)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for param in group["params"]:
                    if param.grad is None:
                        continue

                    state = self.state[param]
                    if len(state) == 0:
                        state["step"] = 0
                        if group["use_quantized_buffers"]:
                            state["momentum_buffer"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), weights_dtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], use_stochastic_rounding=group["use_stochastic_buffers"])
                            if group["adaptive"]:
                                state["v_buffer"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), weights_dtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], use_stochastic_rounding=group["use_stochastic_buffers"])
                        else:
                            state["momentum_buffer"] = torch.zeros_like(param)
                            if group["adaptive"]:
                                state["v_buffer"] = torch.zeros_like(param)

                    state["step"] += 1
                    param_fp32, grad = get_param_grad(param, clip=group["clip_threshold"][0], grad_scale=grad_scale)

                    update = muon_update(
                        param=param_fp32,
                        grad=grad,
                        momentum_buffer=state["momentum_buffer"],
                        v_buffer=state["v_buffer"] if group["adaptive"] else None,
                        step=state["step"],
                        betas=group["betas"],
                        clip=group["clip_threshold"][0],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                        zeropower_dtype=group["zeropower_dtype"],
                        use_quantized_matmul=group["use_quantized_matmul"],
                        quantized_matmul_dtype=group["quantized_matmul_dtype"],
                        use_stochastic_buffers=group["use_stochastic_buffers"],
                    ).to(dtype=torch.float32)

                    update_param_(
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
            else:
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
                    param_fp32, grad = get_param_grad(param, clip=group["clip_threshold"][0], grad_scale=grad_scale)

                    update = adam_update(
                        grad=grad,
                        exp_avg=state["exp_avg"],
                        exp_avg_sq=state["exp_avg_sq"],
                        step=state["step"],
                        betas=group["betas"],
                        clip=group["clip_threshold"][0],
                        use_stochastic_buffers=group["use_stochastic_buffers"],
                    ).to(dtype=torch.float32)

                    update_param_(
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


def muon_update(
    param: torch.FloatTensor,
    grad: torch.FloatTensor,
    momentum_buffer: torch.FloatTensor,
    v_buffer: Optional[torch.FloatTensor],
    step: int,
    betas: Tuple[float, float],
    clip: float,
    ns_steps: int = 5,
    nesterov: bool = True,
    zeropower_dtype: torch.dtype = torch.bfloat16,
    use_quantized_matmul: bool = False,
    quantized_matmul_dtype: str = "int8",
    use_stochastic_buffers: bool = False,
) -> torch.FloatTensor:
    beta1, beta2 = betas
    reshape_grad = (grad.ndim > 2)

    momentum_buffer, momentum_buffer_fp32 = lerp_buffer_stochastic_(momentum_buffer, grad, 1 - beta1, use_stochastic_rounding=use_stochastic_buffers)
    update = grad.lerp(momentum_buffer_fp32, beta1) if nesterov else momentum_buffer_fp32.clone()

    if v_buffer is not None:
        update = update.sign_()

    if reshape_grad: # for the case of conv filters
        grad_shape = grad.shape
        update = update.flatten(1, -1)

    if use_quantized_matmul:
        if quantized_matmul_dtype == "int8":
            update = zeropower_via_newtonschulz5_quantized_matmul(update, int8_matmul_dynamic, steps=ns_steps, clip=clip)
        elif quantized_matmul_dtype in {"fp8", "float8_e4m3fn"}:
            if use_tensorwise_fp8_matmul:
                update = zeropower_via_newtonschulz5_quantized_matmul(update, fp8_matmul_tensorwise_dynamic, steps=ns_steps, clip=clip)
            else:
                update = zeropower_via_newtonschulz5_fp8_matmul(update, steps=ns_steps, clip=clip)
        elif quantized_matmul_dtype in {"fp16", "float16"}:
            update = zeropower_via_newtonschulz5_quantized_matmul(update, fp16_matmul_dynamic, steps=ns_steps, clip=clip)
        else:
            raise NotImplementedError(f"Quantization type {quantized_matmul_dtype} is not implemented")
    else:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps, clip=clip, dtype=zeropower_dtype)

    if reshape_grad:
        update = update.unflatten(-1, grad_shape[1:])

    if v_buffer is not None:
        v_buffer, v_buffer_fp32 = lerp_buffer_stochastic_(v_buffer, update.square(), 1 - beta2, use_stochastic_rounding=use_stochastic_buffers)
        v_hat = v_buffer_fp32 / (1 - beta2 ** step)
        update = update.mul_(v_hat.rsqrt_())

    update = update.nan_to_num_().clamp_(-clip,clip)
    return update


def zeropower_via_newtonschulz5(G: torch.FloatTensor, steps: int = 5, clip: float = 1.0, dtype: torch.dtype = torch.bfloat16) -> torch.FloatTensor:
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(dtype=torch.float32)
    if G.shape[0] > G.shape[1]:
        X = X.t()

    X = torch.div(X, X.norm()).nan_to_num_().clamp_(-clip,clip)
    X = X.to(dtype=dtype)
    for _ in range(steps):
        A = torch.mm(X, X.t())
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a)

    if G.shape[0] > G.shape[1]:
        X = X.t()
    return X.to(dtype=G.dtype)


def zeropower_via_newtonschulz5_quantized_matmul(G: torch.FloatTensor, mm_func: Callable, steps: int = 5, clip: float = 1.0) -> torch.FloatTensor:
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(dtype=torch.float32)
    if G.shape[0] > G.shape[1]:
        X = X.t()

    X = torch.div(X, X.norm()).nan_to_num_().clamp_(-clip,clip)
    for _ in range(steps):
        A = mm_func(X, X, do_input_reshape=True)
        B = mm_func((A*c), A, bias=(A*b), do_input_reshape=False)
        X = mm_func(B, X, bias=(X*a), do_input_reshape=False)

    if G.shape[0] > G.shape[1]:
        X = X.t()
    return X.to(dtype=G.dtype)


def zeropower_via_newtonschulz5_fp8_matmul(G: torch.FloatTensor, steps: int = 5, clip: float = 1.0) -> torch.FloatTensor:
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(dtype=torch.float32)
    if G.shape[0] > G.shape[1]:
        X = X.t()

    X = torch.div(X, X.norm()).nan_to_num_().clamp_(-clip,clip)
    for _ in range(steps):
        A = fp8_matmul_dynamic(X, X, do_input_reshape=True)
        B = fp8_matmul_dynamic((A*c), A, do_input_reshape=False).add_(A, alpha=b)
        X = fp8_matmul_dynamic(B, X, do_input_reshape=False).add_(X, alpha=a)

    if G.shape[0] > G.shape[1]:
        X = X.t()
    return X.to(dtype=G.dtype)


zeropower_via_newtonschulz5_quantized_matmul = compile_func(zeropower_via_newtonschulz5_quantized_matmul)
zeropower_via_newtonschulz5_fp8_matmul = compile_func(zeropower_via_newtonschulz5_fp8_matmul)
