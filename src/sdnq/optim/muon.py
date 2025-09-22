from typing import Tuple, Optional

import torch

from .stochastic import copy_stochastic_
from sdnq.common import compile_func
from sdnq.training import SDNQTensor

from sdnq.training.layers.linear.linear_int8_dynamic import int8_matmul_dynamic
from sdnq.training.layers.linear.linear_fp8_dynamic import fp8_matmul_dynamic


class Muon(torch.optim.Optimizer):
    def __init__(self, params, **kwargs):
        if isinstance(params, torch.nn.Parameter) or (isinstance(params, list) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 1e-3)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["weight_decay"] = group.get("weight_decay", 0.01)
                group["clip_threshold"] = group.get("clip_threshold", (1.0, 1e-3))
                group["ns_steps"] = group.get("ns_steps", 5)
                group["nesterov"] = group.get("nesterov", True)
                group["adaptive"] = group.get("adaptive", False)
                group["norm_mode"] = group.get("norm_mode", "adamuon" if group["adaptive"] else "muon")
                group["bf16_stochastic_round"] = group.get("bf16_stochastic_round", False)
                group["zeropower_dtype"] = group.get("zeropower_dtype", "bfloat16")
                group["use_quantized_matmul"] = group.get("use_quantized_matmul", False)
                group["quantized_matmul_dtype"] = group.get("quantized_matmul_dtype", "int8")
                group["use_quantized_buffers"] = group.get("use_quantized_buffers", False)
                group["quantized_buffers_dtype"] = group.get("quantized_buffers_dtype", "uint8")
                group["quantized_buffers_group_size"] = group.get("quantized_buffers_group_size", 32)
                group["use_stochastic_quantization"] = group.get("use_stochastic_quantization", True)
                if isinstance(group["zeropower_dtype"], str):
                    group["zeropower_dtype"] = getattr(torch, group["zeropower_dtype"])
                assert set(group.keys()) == set(["params", "lr", "use_muon", "betas", "weight_decay", "clip_threshold", "ns_steps", "nesterov", "adaptive", "norm_mode", "bf16_stochastic_round", "zeropower_dtype", "use_quantized_matmul", "quantized_matmul_dtype", "use_quantized_buffers", "quantized_buffers_dtype", "quantized_buffers_group_size", "use_stochastic_quantization"])
            else:
                group["lr"] = group.get("lr", 1e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["weight_decay"] = group.get("weight_decay", 0.01)
                group["clip_threshold"] = group.get("clip_threshold", 1.0)
                group["bf16_stochastic_round"] = group.get("bf16_stochastic_round", False)
                group["use_quantized_buffers"] = group.get("use_quantized_buffers", False)
                group["quantized_buffers_dtype"] = group.get("quantized_buffers_dtype", "uint8")
                group["quantized_buffers_group_size"] = group.get("quantized_buffers_group_size", 32)
                group["use_stochastic_quantization"] = group.get("use_stochastic_quantization", True)
                assert set(group.keys()) == set(["params", "lr", "use_muon", "betas", "weight_decay", "clip_threshold", "bf16_stochastic_round", "use_quantized_buffers", "quantized_buffers_dtype", "quantized_buffers_group_size", "use_stochastic_quantization"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = 0
                        if group["use_quantized_buffers"]:
                            state["momentum_buffer"] = SDNQTensor.from_float(torch.zeros_like(p, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], sr=group["use_stochastic_quantization"])
                            if group["adaptive"]:
                                state["v_buffer"] = SDNQTensor.from_float(torch.zeros_like(p, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], sr=group["use_stochastic_quantization"])
                        else:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            if group["adaptive"]:
                                state["v_buffer"] = torch.zeros_like(p)

                    state["step"] += 1
                    update = muon_update(
                        p,
                        p.grad.to(dtype=state["momentum_buffer"].dtype),
                        state["momentum_buffer"],
                        state["v_buffer"] if group["adaptive"] else None,
                        state["step"],
                        group["betas"],
                        group["clip_threshold"],
                        ns_steps=group["ns_steps"],
                        nesterov=group["nesterov"],
                        norm_mode=group["norm_mode"],
                        zeropower_dtype=group["zeropower_dtype"],
                        use_quantized_matmul=group["use_quantized_matmul"],
                        quantized_matmul_dtype=group["quantized_matmul_dtype"],
                    )

                    if group["bf16_stochastic_round"]:
                        p_fp32 = p.to(dtype=torch.float32)
                        if group["weight_decay"] != 0:
                            p_fp32.mul_(1 - group["lr"] * group["weight_decay"])
                        p_fp32.add_(update, alpha=-group["lr"])
                        copy_stochastic_(p, p_fp32)
                    else:
                        if group["weight_decay"] != 0:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]
                    if len(state) == 0:
                        if group["use_quantized_buffers"]:
                            state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(p, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], sr=group["use_stochastic_quantization"])
                            state["exp_avg_sq"] = SDNQTensor.from_float(torch.zeros_like(p, dtype=torch.float32), qtype=group["quantized_buffers_dtype"], group_size=group["quantized_buffers_group_size"], sr=group["use_stochastic_quantization"])
                        else:
                            state["exp_avg"] = torch.zeros_like(p)
                            state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0

                    state["step"] += 1
                    update = adam_update(
                        p.grad.to(dtype=state["exp_avg"].dtype),
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["clip_threshold"],
                    )

                    if group["bf16_stochastic_round"]:
                        p_fp32 = p.to(dtype=torch.float32)
                        if group["weight_decay"] != 0:
                            p_fp32.mul_(1 - group["lr"] * group["weight_decay"])
                        p_fp32.add_(update, alpha=-group["lr"])
                        copy_stochastic_(p, p_fp32)
                    else:
                        if group["weight_decay"] != 0:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"])

        return loss


def adam_update(grad: torch.FloatTensor, buf1: torch.FloatTensor, buf2: torch.FloatTensor, step: int, betas: Tuple[float, float], clip: float) -> torch.FloatTensor:
    beta, beta2 = betas
    buf1.lerp_(grad, 1 - beta)
    buf2.lerp_(grad.square(), 1 - beta2)
    buf1c = buf1 / (1 - beta ** step)
    buf2c = buf2 / (1 - beta2 ** step)
    return buf1c.mul_(buf2c.rsqrt_()).nan_to_num_().clamp_(-clip,clip)


def muon_update(
    param: torch.FloatTensor,
    grad: torch.FloatTensor,
    momentum_buffer: torch.FloatTensor,
    v_buffer: Optional[torch.FloatTensor],
    step: int,
    betas: Tuple[float, float],
    clips: Tuple[float, float],
    ns_steps: int = 5,
    nesterov: bool = True,
    norm_mode: str = "muon",
    zeropower_dtype: torch.dtype = torch.bfloat16,
    use_quantized_matmul: bool = False,
    quantized_matmul_dtype: str = "int8",
) -> torch.FloatTensor:
    beta, beta2 = betas
    clip, clip2 = clips
    reshape_grad = (grad.ndim > 2)
    momentum_buffer.lerp_(grad, 1 - beta)
    grad = grad.lerp_(momentum_buffer, beta) if nesterov else momentum_buffer

    if reshape_grad: # for the case of conv filters
        grad_shape = grad.shape
        grad = grad.flatten(1, -1)
    output_shape, input_shape = grad.shape

    if use_quantized_matmul:
        if quantized_matmul_dtype == "int8":
            grad = zeropower_via_newtonschulz5_int8_matmul(grad, steps=ns_steps, dtype=zeropower_dtype)
        elif quantized_matmul_dtype == "fp8":
            grad = zeropower_via_newtonschulz5_fp8_matmul(grad, steps=ns_steps, dtype=zeropower_dtype)
        else:
            raise NotImplementedError(f'Quantization type {quantized_matmul_dtype} is not implemented')
    else:
        grad = zeropower_via_newtonschulz5(grad, steps=ns_steps, dtype=zeropower_dtype)

    if reshape_grad:
        grad = grad.unflatten(-1, grad_shape[1:])

    if v_buffer is not None:
        v_buffer.lerp_(grad.square(), 1 - beta2)
        v_hat = v_buffer / (1 - beta2 ** step)
        grad = grad.mul_(v_hat.rsqrt_())
    grad = grad.nan_to_num_().clamp_(-clip,clip)

    if norm_mode == "muon":
        grad = grad.mul_(max(1, output_shape / input_shape)**0.5)
    elif norm_mode == "adamuon":
        grad = grad.mul_(torch.div((clip * 0.2 * grad.numel()**0.5), grad.norm(2)).clamp_(max=1))
    elif norm_mode == "adafactor":
        grad = grad.mul_(param.to(dtype=grad.dtype).norm(2).clamp_(min=clip2).div_(grad.norm(2).clamp_(min=1/clip)))
    elif norm_mode != "none":
        raise NotImplementedError(f'Norm mode {norm_mode} is not implemented')

    return grad


def zeropower_via_newtonschulz5(G: torch.FloatTensor, steps: int = 5, dtype: torch.dtype = torch.bfloat16) -> torch.FloatTensor:
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(dtype=dtype)
    if G.shape[0] > G.shape[1]:
        X = X.t()

    X = torch.div(X, X.norm().clamp_(min=1))
    for _ in range(steps):
        A = torch.mm(X, X.t())
        B = torch.addmm(A, A, A, beta=b, alpha=c)
        X = torch.addmm(X, B, X, beta=a)

    if G.shape[0] > G.shape[1]:
        X = X.t()
    return X.to(dtype=G.dtype)


def zeropower_via_newtonschulz5_int8_matmul(G: torch.FloatTensor, steps: int = 5, dtype: torch.dtype = torch.bfloat16) -> torch.FloatTensor:
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(dtype=dtype)
    if G.shape[0] > G.shape[1]:
        X = X.t()

    X = torch.div(X, X.norm().clamp_(min=1))
    for _ in range(steps):
        A = int8_matmul_dynamic(X, X, None, do_input_reshape=True)
        B = int8_matmul_dynamic((A*c), A, (A*b), do_input_reshape=False)
        X = int8_matmul_dynamic(B, X, (X*a), do_input_reshape=False)

    if G.shape[0] > G.shape[1]:
        X = X.t()
    return X.to(dtype=G.dtype)


def zeropower_via_newtonschulz5_fp8_matmul(G: torch.FloatTensor, steps: int = 5, dtype: torch.dtype = torch.bfloat16) -> torch.FloatTensor:
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(dtype=dtype)
    if G.shape[0] > G.shape[1]:
        X = X.t()

    X = torch.div(X, X.norm().clamp_(min=1))
    for _ in range(steps):
        A = fp8_matmul_dynamic(X, X, None, do_input_reshape=True)
        B = fp8_matmul_dynamic((A*c), A, None, do_input_reshape=False).add_(A, alpha=b)
        X = fp8_matmul_dynamic(B, X, None, do_input_reshape=False).add_(X, alpha=a)

    if G.shape[0] > G.shape[1]:
        X = X.t()
    return X.to(dtype=G.dtype)


zeropower_via_newtonschulz5_int8_matmul = compile_func(zeropower_via_newtonschulz5_int8_matmul)
zeropower_via_newtonschulz5_fp8_matmul = compile_func(zeropower_via_newtonschulz5_fp8_matmul)
