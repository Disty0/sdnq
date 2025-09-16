import torch
import torch.optim

from .stochastic import copy_stochastic_
from sdnq.training import SDNQTensor


class CAME(torch.optim.Optimizer):
    """Implements CAME algorithm.
    This implementation is based on:
    `CAME: Confidence-guided Adaptive Memory Efficient Optimization`
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constants for square gradient
            and instability respectively (default: (1e-30, 1e-16))
        betas (tuple[float, float, float]): coefficient used for computing running averages of
            update, square gradient and instability (default: (0.9, 0.999, 0.9999)))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        clip_threshold (float): threshold of root-mean-square of final gradient update (default: 1.0)
        bf16_stochastic_round (bool): enable or disable BF16 stochastic rounding
    """

    def __init__(self, params, **kwargs):
        if isinstance(params, torch.nn.Parameter) or (isinstance(params, list) and isinstance(params[0], torch.nn.Parameter)):
            kwargs["params"] = params
            param_groups = [kwargs,]
        else:
            param_groups = params
        for group in param_groups:
            group["lr"] = group.get("lr", 1e-4)
            group["eps"] = group.get("eps", (1e-30, 1e-16))
            group["betas"] = group.get("betas", (0.9, 0.999, 0.9999))
            group["weight_decay"] = group.get("weight_decay", 0.0)
            group["clip_threshold"] = group.get("clip_threshold", 1.0)
            group["bf16_stochastic_round"] = group.get("bf16_stochastic_round", False)
            group["use_quantized_buffers"] = group.get("use_quantized_buffers", False)
            group["quantized_buffers_dtype"] = group.get("quantized_buffers_dtype", "int8")
            group["use_stochastic_quantization"] = group.get("use_stochastic_quantization", True)
            assert set(group.keys()) == set(["params", "lr", "eps", "betas", "weight_decay", "clip_threshold", "bf16_stochastic_round", "use_quantized_buffers", "quantized_buffers_dtype", "use_stochastic_quantization"])
        super().__init__(param_groups, dict())

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = torch.div(exp_avg_sq_row, exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored = len(grad_shape) >= 2
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0
                    if group["use_quantized_buffers"]:
                        state["exp_avg"] = SDNQTensor.from_float(torch.zeros_like(grad), qtype=group["quantized_buffers_dtype"], sr=group["use_stochastic_quantization"])
                    else:
                        state["exp_avg"] = torch.zeros_like(grad)

                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device)
                        state["exp_avg_res_row"] = torch.zeros(grad_shape[:-1], dtype=grad.dtype, device=grad.device)
                        state["exp_avg_res_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:], dtype=grad.dtype, device=grad.device)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                state["step"] += 1
                one_minus_betas_1 = 1 - group["betas"][1]
                update = torch.square(grad).add_(group["eps"][0])
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.lerp_(update.mean(dim=-1), one_minus_betas_1)
                    exp_avg_sq_col.lerp_(update.mean(dim=-2), one_minus_betas_1)

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col).mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.lerp_(update, one_minus_betas_1)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(update.norm(2).div_((update.numel() ** 0.5) * group["clip_threshold"]).clamp_(min=1.0))

                exp_avg = state["exp_avg"]
                exp_avg.lerp_(update, 1 - group["betas"][0])


                if factored:
                    # Confidence-guided strategy
                    # Calculation of instability
                    res = torch.sub(update, exp_avg).square_().add_(group["eps"][1])

                    one_minus_betas_2 = 1 - group["betas"][2]
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]
                    exp_avg_res_row.lerp_(res.mean(dim=-1), one_minus_betas_2)
                    exp_avg_res_col.lerp_(res.mean(dim=-2), one_minus_betas_2)

                    # Approximation of exponential moving average of instability
                    update = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col).mul_(exp_avg)
                else:
                    update = exp_avg

                if group["bf16_stochastic_round"]:
                    p_fp32 = p.to(torch.float32)
                    if group["weight_decay"] != 0:
                        p_fp32.mul_(1 - group["lr"] * group["weight_decay"])
                    p_fp32.add_(update, alpha=-group["lr"])
                    copy_stochastic_(p, p_fp32)
                else:
                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
