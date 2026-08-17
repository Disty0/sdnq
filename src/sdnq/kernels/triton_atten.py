import os
import math
import torch
import triton
import triton.language as tl

from ..common import compile_func
from ..quant_utils import quantize_int_mm, quantize_fp_mm, apply_hadamard, get_hadamard, get_hadamard_group_size, rotate_hadamard, rotate_hadamard_compiled
from ..utils import is_pow2, next_power_of_2, get_cache_sizes


min_block_size = int(os.environ.get("SDNQ_TRITON_ATTEN_MIN_BLOCK_SIZE", "256"))
autotune_configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN}, num_warps=w, num_stages=s)
    for BM in [int(BM) for BM in os.environ.get("SDNQ_TRITON_ATTEN_BLOCK_SIZE_M_LIST", "64,128").replace(" ","").split(",")]
    for BN in [int(BN) for BN in os.environ.get("SDNQ_TRITON_ATTEN_BLOCK_SIZE_N_LIST", "16,32").replace(" ","").split(",")]
    for w in [int(w) for w in os.environ.get("SDNQ_TRITON_ATTEN_NUM_WARPS_LIST", "8,16" if torch.xpu.is_available() else "4,8").replace(" ","").split(",")]
    for s in [int(s) for s in os.environ.get("SDNQ_TRITON_ATTEN_NUM_STAGES_LIST", "1" if (torch.cuda.is_available() and torch.version.hip) else "1,2").replace(" ","").split(",")]
]

small_autotune_configs = [
    triton.Config({"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN}, num_warps=w, num_stages=s)
    for BM in [32,64] for BN in [16,32]
    for w in ([4,8] if torch.xpu.is_available() else [2,4])
    for s in ([1,] if (torch.cuda.is_available() and torch.version.hip) else [2,])
]


def prune_configs(configs: list[triton.Config], named_args: dict, from_small: bool = False, **kwargs): # pylint: disable=unused-argument
    device = named_args["q_ptr"].device
    is_dkv_backward = bool(named_args.get("dk_ptr") is not None or named_args.get("dv_ptr") is not None)
    if device.type == "xpu" and torch.int8 in {named_args["q_ptr"].dtype, named_args["v_ptr"].dtype}:
        is_int8_xpu = True
        pruned_configs = [conf for conf in configs if (conf.kwargs["BLOCK_SIZE_M"] >= 32 and conf.kwargs["BLOCK_SIZE_N"] >= 32)]
        if pruned_configs:
            configs = pruned_configs
    else:
        is_int8_xpu = False

    pruned_configs = [
        conf for conf in configs if (
            conf.kwargs["BLOCK_SIZE_M"] <= max(named_args["QN"], 32 if from_small else 64)
            and conf.kwargs["BLOCK_SIZE_N"] <= max(named_args["KN"], 32 if is_int8_xpu else 16)
            and (is_dkv_backward or named_args["is_causal"] == 0 or conf.kwargs["BLOCK_SIZE_M"] >= conf.kwargs["BLOCK_SIZE_N"])
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

            smem_M = block_size_m * named_args["QHD"] * named_args["q_ptr"].element_size()
            if named_args.get("do_ptr") is not None:
                smem_M += block_size_m * named_args["VHD"] * named_args["do_ptr"].element_size()
            if named_args.get("q_scale_ptr") is not None:
                smem_M += block_size_m * named_args["q_scale_ptr"].element_size()
            if named_args.get("do_scale_ptr") is not None:
                smem_M += block_size_m * named_args["do_scale_ptr"].element_size()
            if named_args.get("lse_ptr") is not None and named_args.get("out_ptr") is None:
                smem_M += block_size_m * named_args["lse_ptr"].element_size()
            if named_args.get("delta_ptr") is not None: # Read in BWD
                smem_M += block_size_m * named_args["delta_ptr"].element_size()

            cache_M = smem_M
            if named_args.get("out_ptr") is not None:
                cache_M += block_size_m * named_args["VHD"] * named_args["out_ptr"].element_size()
            if named_args.get("dq_ptr") is not None:
                cache_M += block_size_m * named_args["QHD"] * named_args["dq_ptr"].element_size()
            if named_args.get("lse_ptr") is not None and named_args.get("out_ptr") is not None:
                cache_M += block_size_m * named_args["lse_ptr"].element_size()

            smem_N = block_size_n * named_args["KHD"] * named_args["k_ptr"].element_size()
            smem_N += block_size_n * named_args["VHD"] * named_args["v_ptr"].element_size()
            if named_args.get("k_scale_ptr") is not None:
                smem_N += block_size_n * named_args["k_scale_ptr"].element_size()
            if named_args.get("v_scale_ptr") is not None:
                smem_N += block_size_n * named_args["v_scale_ptr"].element_size()

            cache_N = smem_N
            if named_args.get("dk_ptr") is not None:
                cache_N += block_size_n * named_args["KHD"] * named_args["dk_ptr"].element_size()
            if named_args.get("dv_ptr") is not None:
                cache_N += block_size_n * named_args["VHD"] * named_args["dv_ptr"].element_size()

            if named_args.get("mask_ptr") is not None:
                size_MN = block_size_m * block_size_n * named_args["mask_ptr"].element_size()
            else:
                size_MN = 0
            smem_MN = size_MN
            cache_MN = size_MN

            if is_dkv_backward:
                smem_req = smem_N + ((smem_M + smem_MN) * config.num_stages)
                cache_req = cache_N + ((cache_M + cache_MN) * config.num_stages)
            else:
                smem_req = smem_M + ((smem_N + smem_MN) * config.num_stages)
                cache_req = cache_M + ((cache_N + cache_MN) * config.num_stages)

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
        "is_causal", "do_mask", "save_lse",
        "QZ", "QH", "QN_AT", "QHD",
        "KZ", "KH", "KN_AT", "KHD",
        "VZ", "VH", "VN_AT", "VHD",
        "qk_is_quantized",
        "pv_is_quantized",
        "q_dtype", "v_dtype",
        "out_dtype", "mask_dtype",
    ],
    prune_configs_by={'early_config_prune': prune_configs},
    cache_results=True,
)
@triton.jit
def sdnq_attn_kernel(
    q_ptr, k_ptr, v_ptr,
    q_scale_ptr, k_scale_ptr, v_scale_ptr,
    out_ptr, lse_ptr, mask_ptr, sm_scale,
    is_causal: tl.constexpr,
    do_mask: tl.constexpr,
    save_lse: tl.constexpr,
    QZ: tl.constexpr, QH: tl.constexpr, QN: tl.constexpr, QHD: tl.constexpr,
    KZ: tl.constexpr, KH: tl.constexpr, KN: tl.constexpr, KHD: tl.constexpr,
    VZ: tl.constexpr, VH: tl.constexpr, VN: tl.constexpr, VHD: tl.constexpr,
    MZ: tl.constexpr, MH: tl.constexpr, MQN: tl.constexpr, MKN: tl.constexpr,
    QN_AT: tl.constexpr, # pylint: disable=unused-argument
    KN_AT: tl.constexpr, # pylint: disable=unused-argument
    VN_AT: tl.constexpr, # pylint: disable=unused-argument
    qk_is_quantized: tl.constexpr,
    pv_is_quantized: tl.constexpr,
    q_dtype: tl.constexpr, # pylint: disable=unused-argument
    v_dtype: tl.constexpr, # pylint: disable=unused-argument
    out_dtype: tl.constexpr, # pylint: disable=unused-argument
    mask_dtype: tl.constexpr, # pylint: disable=unused-argument
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
) -> None:
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    tl.assume(QZ > 0)
    tl.assume(QH > 0)
    tl.assume(QN > 0)
    tl.assume(QHD > 0)
    tl.assume(KZ > 0)
    tl.assume(KH > 0)
    tl.assume(KN > 0)
    tl.assume(KHD > 0)
    tl.assume(VZ > 0)
    tl.assume(VH > 0)
    tl.assume(VN > 0)
    tl.assume(VHD > 0)
    tl.assume(MZ >= 0)
    tl.assume(MH >= 0)
    tl.assume(MQN >= 0)
    tl.assume(MKN >= 0)
    tl.assume(off_h >= 0)
    tl.assume(off_z >= 0)
    tl.assume(start_m >= 0)
    tl.assume(BLOCK_SIZE_M > 0)
    tl.assume(BLOCK_SIZE_N > 0)
    tl.assume(do_mask == 0 or do_mask == 1) # pylint: disable=consider-using-in
    tl.assume(save_lse == 0 or save_lse == 1) # pylint: disable=consider-using-in
    tl.assume(is_causal == 0 or is_causal == 1) # pylint: disable=consider-using-in
    tl.assume(qk_is_quantized == 0 or qk_is_quantized == 1) # pylint: disable=consider-using-in
    tl.assume(pv_is_quantized == 0 or pv_is_quantized == 1) # pylint: disable=consider-using-in

    log2_sm_scale = sm_scale * 1.4426950408889634
    do_k_mask = KN % BLOCK_SIZE_N != 0
    start_m_block = start_m * BLOCK_SIZE_M
    offs_m = start_m_block + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offset_q = off_z * (QN * QH) + off_h * QN
    offset_k = off_z * (KN * KH) + ((off_h * KH) // QH) * KN
    offset_v = off_z * (VN * VH) + ((off_h * VH) // QH) * VN

    q_desc = tl.make_tensor_descriptor(q_ptr + offset_q * QHD, shape=[QN, QHD], strides=[QHD, 1], block_shape=[BLOCK_SIZE_M, QHD])
    k_desc = tl.make_tensor_descriptor(k_ptr + offset_k * KHD, shape=[KN, KHD], strides=[KHD, 1], block_shape=[BLOCK_SIZE_N, KHD])
    v_desc = tl.make_tensor_descriptor(v_ptr + offset_v * VHD, shape=[VN, VHD], strides=[VHD, 1], block_shape=[BLOCK_SIZE_N, VHD])

    if qk_is_quantized:
        q_scale_desc = tl.make_tensor_descriptor(q_scale_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
        k_scale_desc = tl.make_tensor_descriptor(k_scale_ptr + offset_k, shape=[KN], strides=[1,], block_shape=[BLOCK_SIZE_N])
        q_scale = q_scale_desc.load([start_m_block])[:, None]
    if pv_is_quantized:
        v_scale_desc = tl.make_tensor_descriptor(v_scale_ptr + offset_v, shape=[VN], strides=[1,], block_shape=[BLOCK_SIZE_N])
    if do_mask:
        mask_desc = tl.make_tensor_descriptor(mask_ptr + offset_q * MKN, shape=[MQN, MKN],  strides=[MKN, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    q = q_desc.load([start_m_block, 0])
    m_i = tl.full([BLOCK_SIZE_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_SIZE_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_M, VHD], dtype=tl.float32)

    for start_n_idx in tl.range(0, tl.cdiv(KN, BLOCK_SIZE_N)):
        start_n = start_n_idx * BLOCK_SIZE_N
        skip = False
        if is_causal and ((start_m_block + BLOCK_SIZE_M) <= start_n):
            skip = True
        if do_mask:
            mask = mask_desc.load([start_m_block, start_n])
            mask_max = tl.max(mask)
            if mask.dtype == tl.int8:
                mask = mask.to(tl.int1)
            else:
                mask = mask.to(tl.float32)
            if mask.dtype == tl.int1:
                if mask_max == 0:
                    skip = True
            elif mask_max == float("-inf"):
                skip = True
        if not skip:
            k = k_desc.load([start_n, 0]).T
            if qk_is_quantized:
                k_scale = k_scale_desc.load([start_n])[None, :]
                if q.dtype == tl.int8:
                    qk = tl.mul(tl.mul(tl.mul(tl.dot(q, k, out_dtype=tl.int32).to(tl.float32), q_scale), k_scale), log2_sm_scale)
                else:
                    qk = tl.mul(tl.mul(tl.mul(tl.dot(q, k, out_dtype=tl.float32), q_scale), k_scale), log2_sm_scale)
            else:
                qk = tl.mul(tl.dot(q, k, out_dtype=tl.float32), log2_sm_scale)

            if is_causal and start_m_block < (start_n + BLOCK_SIZE_N):
                qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
            if do_mask:
                if mask.dtype == tl.int1:
                    qk = tl.where(mask, qk, float("-inf"))
                else:
                    qk += mask
            if do_k_mask and (start_n + BLOCK_SIZE_N) > KN:
                qk = tl.where(offs_n[None, :] < (KN - start_n), qk, float("-inf"))
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            if do_mask:
                alpha = tl.exp2(tl.where((m_i == float("-inf")) & (m_ij == float("-inf")), 0.0, tl.sub(m_i, m_ij)))
                qk -= tl.where(m_ij == float("-inf"), 0.0, m_ij)[:, None]
            else:
                m_i -= m_ij
                alpha = tl.exp2(m_i)
                qk -= m_ij[:, None]
            p = tl.exp2(qk)
            l_i = tl.fma(l_i, alpha, tl.sum(p, 1))
            acc *= alpha[:, None]

            v = v_desc.load([start_n, 0])
            if pv_is_quantized:
                v_scale = v_scale_desc.load([start_n])[None, :]
                p *= v_scale
                p_scale = tl.max(p, 1)[:, None]
                if v.dtype == tl.int8:
                    p_scale *= 1.0 / 127.0
                    p_scale = tl.where(p_scale <= 2e-38, 1.0, p_scale)
                    p = tl.floor(tl.fma(p, tl.fdiv(1.0, p_scale), 0.5)).to(tl.int8)
                    acc = tl.fma(tl.dot(p, v, out_dtype=tl.int32).to(tl.float32), p_scale, acc)
                else:
                    p_scale *= 1.0 / (65504.0 if v.dtype == tl.float16 else 448.0)
                    p_scale = tl.where(p_scale <= 2e-38, 1.0, p_scale)
                    p = tl.mul(p, tl.fdiv(1.0, p_scale)).to(v.dtype)
                    acc = tl.fma(tl.dot(p, v, out_dtype=tl.float32), p_scale, acc)
            else:
                p = p.to(v.dtype)
                acc = tl.dot(p, v, acc, out_dtype=tl.float32)
            m_i = m_ij

    acc *= tl.fdiv(1.0, l_i[:, None])
    acc = acc.to(out_ptr.type.element_ty)
    out_desc = tl.make_tensor_descriptor(out_ptr + offset_q * VHD, shape=[QN, VHD], strides=[VHD, 1], block_shape=[BLOCK_SIZE_M, VHD])
    out_desc.store([start_m_block, 0], acc)
    if save_lse:
        l_i = tl.add(m_i, tl.log2(l_i))
        if do_mask:
            l_i = tl.where(l_i == float("-inf"), 0.0, l_i)
        l_i = l_i.to(out_ptr.type.element_ty)
        l_desc = tl.make_tensor_descriptor(lse_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
        l_desc.store([start_m_block], l_i)


def quantize_attn(
    q, k, v,
    smooth_k: bool = True,
    hadamard: torch.FloatTensor | None = None,
    hadamard_group_size: int = 256,
    matmul_dtype: str = "int8",
    pv_matmul_dtype: str | None = None,
) -> tuple[torch.Tensor, torch.FloatTensor | None, torch.Tensor, torch.FloatTensor | None, torch.Tensor, torch.FloatTensor | None, bool, int]:
    if matmul_dtype in {"auto", "enabled", "uint8"}:
        matmul_dtype = "int8"
    if pv_matmul_dtype in {"enabled", "uint8"}:
        pv_matmul_dtype = "int8"
    if smooth_k:
        if k.dtype != torch.float32:
            k = k.to(dtype=torch.float32)
            k = k.sub_(k.mean(dim=2, keepdim=True))
        else:
            k = k.sub(k.mean(dim=2, keepdim=True))
    use_hadamard = False
    if matmul_dtype not in {None, "none", "no", "disabled"}:
        if hadamard is not None:
            q, use_hadamard, hadamard_group_size = apply_hadamard(q, group_size=hadamard_group_size, hadamard=hadamard, layer_class_name="Linear")
            if use_hadamard:
                k = rotate_hadamard(k.to(dtype=hadamard.dtype), group_size=hadamard_group_size, hadamard=hadamard)
        quantize_mm_func = quantize_int_mm if matmul_dtype.startswith("int") else quantize_fp_mm
        q_q, q_scale = quantize_mm_func(q.contiguous().to(dtype=torch.float32), dim=-1, matmul_dtype=matmul_dtype)
        k_q, k_scale = quantize_mm_func(k.contiguous().to(dtype=torch.float32), dim=-1, matmul_dtype=matmul_dtype)
        q_scale = q_scale.squeeze(-1)
        k_scale = k_scale.squeeze(-1)
    else:
        q_q = q.contiguous()
        k_q = k.contiguous().to(dtype=q.dtype)
        q_scale = None
        k_scale = None
    if pv_matmul_dtype not in {None, "auto", "none", "no", "disabled"}:
        if use_hadamard:
            v = rotate_hadamard(v.to(dtype=hadamard.dtype), group_size=hadamard_group_size, hadamard=hadamard)
        quantize_mm_func_pv = quantize_int_mm if pv_matmul_dtype.startswith("int") else quantize_fp_mm
        v_q, v_scale = quantize_mm_func_pv(v.contiguous().to(dtype=torch.float32), dim=-1, matmul_dtype=pv_matmul_dtype)
        v_scale = v_scale.squeeze(-1)
    else:
        v_q = v.contiguous()
        v_scale = None
    return q_q, q_scale, k_q, k_scale, v_q, v_scale, use_hadamard, hadamard_group_size


def get_attn_inputs(
    query: torch.FloatTensor,
    key: torch.FloatTensor,
    value: torch.FloatTensor,
    hadamard: torch.FloatTensor | None = None,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0, # pylint: disable=unused-argument
    is_causal: bool = False, # pylint: disable=unused-argument
    scale: float | None = None,
    enable_gqa: bool = False, # pylint: disable=unused-argument
    smooth_k: bool = True,
    hadamard_group_size: int = 256,
    matmul_dtype: str = "int8",
    pv_matmul_dtype: str | None = None,
    do_quantize: bool = True,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.FloatTensor | None, torch.Tensor, torch.FloatTensor | None, torch.Tensor, torch.FloatTensor | None, torch.Tensor | None, float, torch.dtype, bool, int]:
    QZ, QH, QN, QHD = query.shape
    _, _, KN, KHD = key.shape
    _, _, _, VHD = value.shape
    if out_dtype is None:
        out_dtype = query.dtype
    if scale is None:
        scale = QHD ** -0.5
    if not is_pow2(QHD):
        query = torch.nn.functional.pad(query, (0, next_power_of_2(QHD) - QHD))
        key = torch.nn.functional.pad(key, (0, next_power_of_2(KHD) - KHD))
        value = torch.nn.functional.pad(value, (0, next_power_of_2(VHD) - VHD))
    if attn_mask is not None:
        attn_mask = attn_mask.expand((QZ, QH, QN, KN))
        if not is_pow2(KN):
            pad_value = float("-inf") if torch.is_floating_point(attn_mask) else 0
            attn_mask = torch.nn.functional.pad(attn_mask, (0, next_power_of_2(KN) - KN), value=pad_value)
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.to(dtype=torch.int8)
        attn_mask = attn_mask.contiguous()
    query, query_scale, key, key_scale, value, value_scale, use_hadamard, hadamard_group_size = quantize_attn(
        query, key, value,
        smooth_k=smooth_k,
        hadamard=hadamard,
        hadamard_group_size=hadamard_group_size,
        matmul_dtype=matmul_dtype if do_quantize else "disabled",
        pv_matmul_dtype=pv_matmul_dtype if do_quantize else "disabled",
    )
    return query, query_scale, key, key_scale, value, value_scale, attn_mask, scale, out_dtype, use_hadamard, hadamard_group_size


def sdnq_triton_atten(
    query: torch.FloatTensor,
    key: torch.FloatTensor,
    value: torch.FloatTensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0, # pylint: disable=unused-argument
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False, # pylint: disable=unused-argument
    smooth_k: bool = True,
    use_hadamard: bool = False,
    hadamard_group_size: int = 256,
    matmul_dtype: str = "int8",
    pv_matmul_dtype: str | None = None,
    do_quantize: bool = True,
    out_dtype: torch.dtype | None = None,
    return_backward: bool = False,
) -> torch.FloatTensor:
    QZ, QH, QN, QHD = query.shape
    _, _, KN, KHD = key.shape
    _, _, VN, VHD = value.shape

    hadamard = None
    if use_hadamard and do_quantize and matmul_dtype not in {None, "none", "no", "disabled"}:
        hadamard_channel_size = next_power_of_2(min(QHD, KHD))
        hadamard_group_size = min(hadamard_group_size, hadamard_channel_size)
        use_hadamard, hadamard_group_size = get_hadamard_group_size(hadamard_channel_size, hadamard_group_size)
        if use_hadamard:
            hadamard = get_hadamard(hadamard_group_size, dtype=query.dtype, device=query.device)

    (
        query, query_scale,
        key, key_scale,
        value, value_scale,
        attn_mask, scale, out_dtype,
        use_hadamard, hadamard_group_size,
    ) = get_attn_inputs(
        query=query, key=key, value=value,
        hadamard=hadamard, attn_mask=attn_mask,
        dropout_p=dropout_p, is_causal=is_causal,
        scale=scale, enable_gqa=enable_gqa,
        smooth_k=smooth_k, hadamard_group_size=hadamard_group_size,
        matmul_dtype=matmul_dtype, pv_matmul_dtype=pv_matmul_dtype,
        do_quantize=do_quantize, out_dtype=out_dtype,
    )

    def grid(META):
        return (triton.cdiv(QN, META["BLOCK_SIZE_M"]), QH, QZ)
    out = torch.empty((QZ, QH, QN, value.shape[-1]), dtype=out_dtype, device=query.device)
    lse = torch.empty((QZ, QH, QN), dtype=out_dtype, device=query.device) if return_backward else None

    sdnq_attn_kernel[grid](
        query, key, value,
        query_scale, key_scale, value_scale,
        out, lse, attn_mask, scale,
        (1 if is_causal else 0),
        (1 if attn_mask is not None else 0),
        (1 if return_backward else 0),
        *query.shape, *key.shape, *value.shape,
        *(attn_mask.shape if attn_mask is not None else (0, 0, 0, 0)),
        math.ceil(QN / min_block_size),
        math.ceil(KN / min_block_size),
        math.ceil(VN / min_block_size),
        (1 if query_scale is not None else 0),
        (1 if value_scale is not None else 0),
        str(query.dtype), str(value.dtype), str(out.dtype),
        str(attn_mask.dtype if attn_mask is not None else None),
    )

    if use_hadamard and pv_matmul_dtype not in {None, "auto", "none", "no", "disabled"}:
        if hadamard.shape[-1] != hadamard_group_size:
            hadamard = get_hadamard(hadamard_group_size, dtype=out.dtype, device=out.device)
        out = rotate_hadamard_compiled(out, group_size=hadamard_group_size, hadamard=hadamard)

    if return_backward:
        return out, lse, query, key, value, query_scale, key_scale, value_scale, attn_mask, scale, use_hadamard, hadamard_group_size
    return out[..., :VHD]


get_attn_inputs = compile_func(get_attn_inputs)
