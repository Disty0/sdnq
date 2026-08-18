import math

import torch
from torch.library import triton_op, wrap_triton

import triton
import triton.language as tl

from ..sdnext import devices
from .triton_scaled_mm import min_block_size, autotune_configs, prune_configs, USE_FP16_ACCUM


@triton.autotune(
    configs=autotune_configs,
    key=[
        "bias_ndim",
        "b_is_contiguous",
        "use_fp16_accum",
        "M_AT", "N_AT", "K_AT",
        "a_dtype", "out_dtype",
    ],
    prune_configs_by={'early_config_prune': prune_configs},
    cache_results=True,
)
@triton.jit
def sdnq_triton_mm_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    bias_ndim: tl.constexpr,
    b_is_contiguous: tl.constexpr,
    use_fp16_accum: tl.constexpr,
    M_AT: tl.constexpr, # pylint: disable=unused-argument
    N_AT: tl.constexpr, # pylint: disable=unused-argument
    K_AT: tl.constexpr, # pylint: disable=unused-argument
    a_dtype: tl.constexpr, # pylint: disable=unused-argument
    out_dtype: tl.constexpr, # pylint: disable=unused-argument
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
) -> None:
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    off_m = pid_m * BLOCK_SIZE_M
    off_n = pid_n * BLOCK_SIZE_N

    tl.assume(M > 0)
    tl.assume(N > 0)
    tl.assume(K > 0)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(off_m >= 0)
    tl.assume(off_n >= 0)
    tl.assume(BLOCK_SIZE_M > 0)
    tl.assume(BLOCK_SIZE_N > 0)
    tl.assume(BLOCK_SIZE_K > 0)
    tl.assume(GROUP_SIZE_M > 0)
    tl.assume(bias_ndim >= 0 and bias_ndim <= 2) # pylint: disable=consider-using-in
    tl.assume(b_is_contiguous == 0 or b_is_contiguous == 1) # pylint: disable=consider-using-in
    tl.assume(use_fp16_accum == 0 or use_fp16_accum == 1) # pylint: disable=consider-using-in

    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(K, 1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
    if b_is_contiguous:
        b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(K, N), strides=(N, 1), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))
    else:
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_bn = (off_n + tl.arange(0, BLOCK_SIZE_N)) % N
        b_ptrs = b_ptr + (offs_k[:, None] + offs_bn[None, :] * K)

    if use_fp16_accum and a_ptr.type.element_ty == tl.float16:
        fp16_scale = 65536.0 * K
        in_scale = 1.0 / (65536.0 * K)**0.5
        accumulator_dtype = tl.float16
    else:
        accumulator_dtype = tl.int32 if a_ptr.type.element_ty == tl.int8 else tl.float32

    off_k = 0
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)
    for _ in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load([off_m, off_k])
        if b_is_contiguous:
            b = b_desc.load([off_k, off_n])
        else:
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - off_k, other=0.0)
            b_ptrs += BLOCK_SIZE_K
        if use_fp16_accum and a_ptr.type.element_ty == tl.float16:
            a = tl.mul(a.to(tl.float32), in_scale).to(tl.float16)
            b = tl.mul(b.to(tl.float32), in_scale).to(tl.float16)
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)
        off_k += BLOCK_SIZE_K
    if use_fp16_accum and a_ptr.type.element_ty == tl.float16:
        accumulator = tl.mul(accumulator.to(tl.float32), fp16_scale)

    if bias_ndim == 1:
        accumulator = accumulator.to(tl.float32)
        bias_desc = tl.make_tensor_descriptor(base=bias_ptr, shape=(N,), strides=(1,), block_shape=(BLOCK_SIZE_N,))
        bias = bias_desc.load([off_n])[None, :].to(tl.float32)
        accumulator += bias
    elif bias_ndim == 2:
        accumulator = accumulator.to(tl.float32)
        bias_desc = tl.make_tensor_descriptor(base=bias_ptr, shape=(M, N), strides=(N, 1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
        bias = bias_desc.load([off_m, off_n]).to(tl.float32)
        accumulator += bias

    accumulator = accumulator.to(c_ptr.type.element_ty)
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(N, 1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([off_m, off_n], accumulator)


@triton_op("sdnq::triton_mm", mutates_args={})
@devices.inference_context()
def sdnq_triton_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    if bias is not None:
        assert bias.is_contiguous(), "Bias must be contiguous"
        assert bias.ndim in {1, 2}, "Bias must be 1D or 2D"
    M, K = a.shape
    K, N = b.shape
    if out_dtype is None:
        out_dtype = torch.int32 if a.dtype == torch.int8 else torch.float32
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    wrap_triton(sdnq_triton_mm_kernel)[grid](
        a, b, c, bias,
        M, N, K,
        (0 if bias is None else bias.ndim),
        (1 if b.is_contiguous() else 0),
        (1 if USE_FP16_ACCUM else 0),
        math.ceil(M / min_block_size),
        math.ceil(N / min_block_size),
        math.ceil(K / min_block_size),
        str(a.dtype), str(c.dtype),
    )
    return c
