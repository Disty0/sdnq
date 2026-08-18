import os
import math

import torch
from torch.library import triton_op, wrap_triton

import triton
import triton.language as tl

from ..sdnext import devices
from ..utils import get_cache_sizes


USE_FP16_ACCUM = bool(os.environ.get("SDNQ_TRITON_MM_USE_FP16_ACCUM", "0").lower() not in {"0", "false", "no"})

min_block_size = int(os.environ.get("SDNQ_TRITON_MM_MIN_BLOCK_SIZE", "256"))
autotune_configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GM}, num_warps=w, num_stages=s)
    for BM in [int(BM) for BM in os.environ.get("SDNQ_TRITON_MM_BLOCK_SIZE_M_LIST", "64,128").replace(" ","").split(",")]
    for BN in [int(BN) for BN in os.environ.get("SDNQ_TRITON_MM_BLOCK_SIZE_N_LIST", "64,128,256").replace(" ","").split(",")]
    for BK in [int(BK) for BK in os.environ.get("SDNQ_TRITON_MM_BLOCK_SIZE_K_LIST", "32,64,128").replace(" ","").split(",")]
    for GM in [int(GM) for GM in os.environ.get("SDNQ_TRITON_MM_GROUP_SIZE_M_LIST", "8").replace(" ","").split(",")]
    for w in [int(w) for w in os.environ.get("SDNQ_TRITON_MM_NUM_WARPS_LIST", "16" if torch.xpu.is_available() else "4").replace(" ","").split(",")]
    for s in [int(s) for s in os.environ.get("SDNQ_TRITON_MM_NUM_STAGES_LIST", "1" if (torch.cuda.is_available() and torch.version.hip) else "2").replace(" ","").split(",")]
]

small_autotune_configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": GM}, num_warps=w, num_stages=s)
    for BM in [32,64] for BN in [32,64,128] for BK in [16,32,64] for GM in [4,]
    for w in ([8,] if torch.xpu.is_available() else [2,])
    for s in ([1,] if (torch.cuda.is_available() and torch.version.hip) else [2,])
]


def prune_configs(configs: list[triton.Config], named_args: dict, from_small: bool = False, **kwargs): # pylint: disable=unused-argument
    device = named_args["a_ptr"].device
    if device.type == "xpu" and named_args["a_ptr"].dtype == torch.int8:
        is_int8_xpu = True
        pruned_configs = [conf for conf in configs if (conf.kwargs["BLOCK_SIZE_M"] >= 32 and conf.kwargs["BLOCK_SIZE_N"] >= 32 and conf.kwargs["BLOCK_SIZE_K"] >= 32)]
        if pruned_configs:
            configs = pruned_configs
    else:
        is_int8_xpu = False

    pruned_configs = [
        conf for conf in configs if (
            conf.kwargs["BLOCK_SIZE_M"] <= max(named_args["M"], 32 if from_small else 64)
            and conf.kwargs["BLOCK_SIZE_N"] <= max(named_args["N"], 32 if from_small else 64)
            and conf.kwargs["BLOCK_SIZE_K"] <= max(named_args["K"], 16 if (from_small and not is_int8_xpu) else 32)
        )
    ]
    if pruned_configs:
        configs = pruned_configs

    cache_size, smem_size = get_cache_sizes(device)
    if cache_size > 0 or smem_size > 0:
        pruned_configs = []
        for config in configs:
            block_size_m = config.kwargs["BLOCK_SIZE_M"]
            block_size_n = config.kwargs["BLOCK_SIZE_N"]
            block_size_k = config.kwargs["BLOCK_SIZE_K"]
            group_size_m = config.kwargs["GROUP_SIZE_M"]

            if smem_size > 0:
                smem_req = block_size_m * block_size_k * named_args["a_ptr"].element_size()
                smem_req += block_size_n * block_size_k * named_args["b_ptr"].element_size()
                smem_req *= config.num_stages
            else:
                smem_req = 0

            if cache_size > 0:
                cache_req = group_size_m * block_size_m * block_size_k * named_args["a_ptr"].element_size()
                cache_req += block_size_n * block_size_k * named_args["b_ptr"].element_size()
                cache_req *= config.num_stages

                cache_req += group_size_m * block_size_m * block_size_n * named_args["c_ptr"].element_size()
                if named_args.get("scale_a_ptr") is not None:
                    cache_req += group_size_m * block_size_m * named_args["scale_a_ptr"].element_size()
                if named_args.get("scale_b_ptr") is not None:
                    cache_req += block_size_n * named_args["scale_b_ptr"].element_size()
                if named_args.get("bias_ptr") is not None:
                    if named_args["bias_ndim"] == 1:
                        cache_req += block_size_n * named_args["bias_ptr"].element_size()
                    else:
                        cache_req += group_size_m * block_size_m * block_size_n * named_args["bias_ptr"].element_size()
            else:
                cache_req = 0

            if (cache_req <= cache_size or cache_size == 0) and (smem_req <= smem_size or smem_size == 0):
                pruned_configs.append(config)

        if pruned_configs:
            configs = pruned_configs
        elif not from_small:
            return prune_configs(small_autotune_configs, named_args, from_small=True, **kwargs)
    return configs


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
def sdnq_scaled_mm_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    scale_a_ptr, scale_b_ptr,
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

    scale_a_desc = tl.make_tensor_descriptor(base=scale_a_ptr, shape=(M,), strides=(1,), block_shape=(BLOCK_SIZE_M,))
    scale_b_desc = tl.make_tensor_descriptor(base=scale_b_ptr, shape=(N,), strides=(1,), block_shape=(BLOCK_SIZE_N,))
    scale_a = scale_a_desc.load([off_m])[:, None].to(tl.float32)
    scale_b = scale_b_desc.load([off_n])[None, :].to(tl.float32)

    if bias_ndim == 1:
        accumulator = tl.mul(accumulator.to(tl.float32), scale_a)
        bias_desc = tl.make_tensor_descriptor(base=bias_ptr, shape=(N,), strides=(1,), block_shape=(BLOCK_SIZE_N,))
        bias = bias_desc.load([off_n])[None, :].to(tl.float32)
        accumulator = tl.fma(accumulator, scale_b, bias)
    elif bias_ndim == 2:
        accumulator = tl.mul(accumulator.to(tl.float32), scale_a)
        bias_desc = tl.make_tensor_descriptor(base=bias_ptr, shape=(M, N), strides=(N, 1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
        bias = bias_desc.load([off_m, off_n]).to(tl.float32)
        accumulator = tl.fma(accumulator, scale_b, bias)
    else:
        accumulator = tl.mul(tl.mul(accumulator.to(tl.float32), scale_a), scale_b)

    accumulator = accumulator.to(c_ptr.type.element_ty)
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(N, 1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([off_m, off_n], accumulator)


@triton_op("sdnq::scaled_mm", mutates_args={})
@devices.inference_context()
def sdnq_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert scale_a.is_contiguous(), "Matrix A scale must be contiguous"
    assert scale_b.is_contiguous(), "Matrix B scale must be contiguous"
    if bias is not None:
        assert bias.is_contiguous(), "Bias must be contiguous"
        assert bias.ndim in {1, 2}, "Bias must be 1D or 2D"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    wrap_triton(sdnq_scaled_mm_kernel)[grid](
        a, b, c, bias,
        scale_a, scale_b,
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
