from typing import Tuple, Optional, Iterator

import torch

from .optimizer import SDNQOptimizer
from sdnq.training import SDNQTensor

from sdnq.common import compile_func
from sdnq.training.layers.linear.linear_int8_dynamic import int8_matmul_dynamic
from sdnq.training.layers.linear.linear_fp8_dynamic import fp8_matmul_dynamic


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
        if isinstance(params, (torch.nn.Parameter, Iterator)) or (isinstance(params, list) and isinstance(params[0], torch.nn.Parameter)):
            muon_group = {"use_muon": True, "params": []}
            adamw_group = {"use_muon": False, "params": []}
            keys_to_pop = []
            for key, value in kwargs.items():
                if key.startswith("muon_"):
                    muon_group[key.removeprefix("muon_")] = value
                    keys_to_pop.append(key)
                elif key.startswith("adamw_"):
                    adamw_group[key.removeprefix("adamw_")] = value
                    keys_to_pop.append(key)
            for key in keys_to_pop:
                kwargs.pop(key, None)
            for key, value in kwargs.items():
                if key not in muon_group.keys():
                    muon_group[key] = value
                if key not in adamw_group.keys() and key not in self._extra_group_keys[0]:
                    adamw_group[key] = value
            for param in params:
                if param.ndim <= 1:
                    adamw_group["params"].append(param)
                else:
                    muon_group["params"].append(param)
            param_groups = [muon_group, adamw_group]
        else:
            param_groups = params
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 1e-3)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["nesterov"] = group.get("nesterov", True)
                group["adaptive"] = group.get("adaptive", False)
                group["final_norm_mode"] = group.get("final_norm_mode", "rms_clip_scaled")
                group["zeropower_dtype"] = group.get("zeropower_dtype", "bfloat16")
                group["use_quantized_matmul"] = group.get("use_quantized_matmul", False)
                group["quantized_matmul_dtype"] = group.get("quantized_matmul_dtype", "int8")
                if isinstance(group["zeropower_dtype"], str):
                    group["zeropower_dtype"] = getattr(torch, group["zeropower_dtype"])
                group = self.apply_group_defaults(group)
                assert set(group.keys()) == self._group_keys[0]
            else:
                group = self.apply_group_defaults(group)
                assert set(group.keys()) == self._group_keys[1]
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        grad_scale = getattr(self, "grad_scale", None)
        found_inf = getattr(self, "found_inf", 0)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for param in group["params"]:
                    if param.grad is None or found_inf > 0:
                        continue

                    state = self.state[param]
                    if len(state) == 0:
                        state["step"] = 0
                        if group["use_quantized_buffers"]:
                            state["momentum_buffer"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], sr=group["use_stochastic_quantization"])
                            if group["adaptive"]:
                                state["v_buffer"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], sr=group["use_stochastic_quantization"])
                        else:
                            state["momentum_buffer"] = torch.zeros_like(param)
                            if group["adaptive"]:
                                state["v_buffer"] = torch.zeros_like(param)

                    state["step"] += 1
                    param_fp32 = param.to(dtype=torch.float32)
                    grad = param.grad.to(dtype=torch.float32)
                    if grad_scale is not None:
                        grad.div_(grad_scale.to(dtype=torch.float32))

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
            else:
                for param in group["params"]:
                    if param.grad is None or found_inf > 0:
                        continue

                    state = self.state[param]
                    if len(state) == 0:
                        state["step"] = 0
                        if group["use_quantized_buffers"]:
                            state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], sr=group["use_stochastic_quantization"])
                            state["exp_avg_sq"] = SDNQTensor.from_float(torch.zeros_like(param, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], svd_rank=group["quantized_buffers_svd_rank"], use_svd=group["use_svd_quantization"], sr=group["use_stochastic_quantization"])
                        else:
                            state["exp_avg"] = torch.zeros_like(param)
                            state["exp_avg_sq"] = torch.zeros_like(param)

                    state["step"] += 1
                    param_fp32 = param.to(dtype=torch.float32)
                    grad = param.grad.to(dtype=torch.float32)
                    if grad_scale is not None:
                        grad.div_(grad_scale.to(dtype=torch.float32))

                    update = adam_update(
                        grad=grad,
                        exp_avg=state["exp_avg"],
                        exp_avg_sq=state["exp_avg_sq"],
                        step=state["step"],
                        betas=group["betas"],
                        clip=group["clip_threshold"][0],
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


def adam_update(grad: torch.FloatTensor, exp_avg: torch.FloatTensor, exp_avg_sq: torch.FloatTensor, step: int, betas: Tuple[float, float], clip: float) -> torch.FloatTensor:
    beta1, beta2 = betas
    if exp_avg.dtype != torch.float32:
        exp_avg_fp32 = exp_avg.to(dtype=torch.float32).lerp_(grad, 1 - beta1)
        exp_avg_sq_fp32 = exp_avg_sq.to(dtype=torch.float32).lerp_(grad.square(), 1 - beta2)
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
) -> torch.FloatTensor:
    beta1, beta2 = betas
    reshape_grad = (grad.ndim > 2)

    if momentum_buffer.dtype != torch.float32:
        momentum_buffer_fp32 = momentum_buffer.to(dtype=torch.float32).lerp_(grad, 1 - beta1)
        momentum_buffer.copy_(momentum_buffer_fp32)
    else:
        momentum_buffer.lerp_(grad, 1 - beta1)
        momentum_buffer_fp32 = momentum_buffer
    update = grad.lerp(momentum_buffer_fp32, beta1) if nesterov else momentum_buffer_fp32.clone()

    if v_buffer is not None:
        update = update.sign_()

    if reshape_grad: # for the case of conv filters
        grad_shape = grad.shape
        update = update.flatten(1, -1)

    if use_quantized_matmul:
        if quantized_matmul_dtype == "int8":
            update = zeropower_via_newtonschulz5_int8_matmul(update, steps=ns_steps, clip=clip)
        elif quantized_matmul_dtype == "fp8":
            update = zeropower_via_newtonschulz5_fp8_matmul(update, steps=ns_steps, clip=clip)
        else:
            raise NotImplementedError(f'Quantization type {quantized_matmul_dtype} is not implemented')
    else:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps, clip=clip, dtype=zeropower_dtype)

    if reshape_grad:
        update = update.unflatten(-1, grad_shape[1:])

    if v_buffer is not None:
        if v_buffer.dtype != torch.float32:
            v_buffer_fp32 = v_buffer.to(dtype=torch.float32).lerp_(update.square(), 1 - beta2)
            v_buffer.copy_(v_buffer_fp32)
        else:
            v_buffer.lerp_(update.square(), 1 - beta2)
            v_buffer_fp32 = v_buffer
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


def zeropower_via_newtonschulz5_int8_matmul(G: torch.FloatTensor, steps: int = 5, clip: float = 1.0) -> torch.FloatTensor:
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(dtype=torch.float32)
    if G.shape[0] > G.shape[1]:
        X = X.t()

    X = torch.div(X, X.norm()).nan_to_num_().clamp_(-clip,clip)
    for _ in range(steps):
        A = int8_matmul_dynamic(X, X, do_input_reshape=True)
        B = int8_matmul_dynamic((A*c), A, bias=(A*b), do_input_reshape=False)
        X = int8_matmul_dynamic(B, X, bias=(X*a), do_input_reshape=False)

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


zeropower_via_newtonschulz5_int8_matmul = compile_func(zeropower_via_newtonschulz5_int8_matmul)
zeropower_via_newtonschulz5_fp8_matmul = compile_func(zeropower_via_newtonschulz5_fp8_matmul)
