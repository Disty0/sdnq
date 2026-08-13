import math
import torch
import triton
import triton.language as tl

from ..common import compile_func
from ..quant_utils import quantize_int_mm, quantize_fp_mm, get_hadamard, apply_hadamard, rotate_hadamard_compiled
from .triton_atten import sdnq_triton_atten, autotune_configs, min_block_size


@triton.autotune(
    configs=autotune_configs,
    key=[
        "is_causal", "do_mask",
        "QZ", "QH", "QN_AT", "QHD",
        "KZ", "KH", "KN_AT", "KHD",
        "VZ", "VH", "VN_AT", "VHD",
        "qk_is_quantized",
        "pv_is_quantized",
        "q_dtype", "v_dtype",
        "out_dtype", "return_dtype",
        "mask_dtype",
    ],
    cache_results=True,
)
@triton.jit
def sdnq_attn_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr,
    q_scale_ptr, k_scale_ptr, v_scale_ptr, do_scale_ptr,
    dq_ptr, lse_ptr, delta_ptr, mask_ptr,
    is_causal: tl.constexpr,
    do_mask: tl.constexpr,
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
    return_dtype: tl.constexpr, # pylint: disable=unused-argument
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
    tl.assume(is_causal == 0 or is_causal == 1) # pylint: disable=consider-using-in
    tl.assume(qk_is_quantized == 0 or qk_is_quantized == 1) # pylint: disable=consider-using-in
    tl.assume(pv_is_quantized == 0 or pv_is_quantized == 1) # pylint: disable=consider-using-in

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
    do_desc = tl.make_tensor_descriptor(do_ptr + offset_q * QHD, shape=[QN, QHD], strides=[QHD, 1], block_shape=[BLOCK_SIZE_M, QHD])
    lse_desc = tl.make_tensor_descriptor(lse_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
    delta_desc = tl.make_tensor_descriptor(delta_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])

    if qk_is_quantized:
        q_scale_desc = tl.make_tensor_descriptor(q_scale_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
        q_scale = q_scale_desc.load([start_m_block])[:, None]
    if pv_is_quantized:
        do_scale_desc = tl.make_tensor_descriptor(do_scale_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
        do_scale = do_scale_desc.load([start_m_block])[:, None]

    q = q_desc.load([start_m_block, 0])
    do = do_desc.load([start_m_block, 0])
    lse = lse_desc.load([start_m_block]).to(tl.float32)
    delta = delta_desc.load([start_m_block]).to(tl.float32)
    dq = tl.zeros([BLOCK_SIZE_M, QHD], dtype=tl.float32)

    if qk_is_quantized:
        k_scale_desc = tl.make_tensor_descriptor(k_scale_ptr + offset_k, shape=[KN], strides=[1,], block_shape=[BLOCK_SIZE_N])
    if pv_is_quantized:
        v_scale_desc = tl.make_tensor_descriptor(v_scale_ptr + offset_v, shape=[VN], strides=[1,], block_shape=[BLOCK_SIZE_N])
    if do_mask:
        mask_desc = tl.make_tensor_descriptor(mask_ptr + offset_q * MKN, shape=[MQN, MKN], strides=[MKN, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

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
            k_load = k_desc.load([start_n, 0])
            k = k_load.T
            if qk_is_quantized:
                k_scale = k_scale_desc.load([start_n])[None, :]
                if q.dtype == tl.int8:
                    qk = tl.mul(tl.mul(tl.dot(q, k, out_dtype=tl.int32).to(tl.float32), q_scale), k_scale)
                else:
                    qk = tl.mul(tl.mul(tl.dot(q, k, out_dtype=tl.float32), q_scale), k_scale)
            else:
                qk = tl.dot(q, k, out_dtype=tl.float32)

            if is_causal and start_m_block < (start_n + BLOCK_SIZE_N):
                qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
            if do_mask:
                if mask.dtype == tl.int1:
                    qk = tl.where(mask, qk, float("-inf"))
                else:
                    qk += mask
            if do_k_mask and (start_n + BLOCK_SIZE_N) > KN:
                qk = tl.where(offs_n[None, :] < (KN - start_n), qk, float("-inf"))

            qk -= lse[:, None]
            p = tl.exp2(qk)

            v = v_desc.load([start_n, 0]).T
            if pv_is_quantized:
                v_scale = v_scale_desc.load([start_n])[None, :]
                if do.dtype == tl.int8:
                    dp = tl.mul(tl.mul(tl.dot(do, v, out_dtype=tl.int32).to(tl.float32), do_scale), v_scale)
                else:
                    dp = tl.mul(tl.mul(tl.dot(do, v, out_dtype=tl.float32), do_scale), v_scale)
            else:
                dp = tl.dot(do, v, out_dtype=tl.float32)

            ds = tl.mul(p, tl.sub(dp, delta[:, None]))
            if qk_is_quantized:
                ds *= k_scale
                ds_scale = tl.max(tl.abs(ds), 1)[:, None]
                if k.dtype == tl.int8:
                    ds_scale *= 1.0 / 127.0
                    ds_scale = tl.where(ds_scale <= 2e-38, 1.0, ds_scale)
                    ds = tl.floor(tl.fma(ds, (1.0 / ds_scale), 0.5)).to(tl.int8)
                    dq = tl.fma(tl.dot(ds, k_load, out_dtype=tl.int32).to(tl.float32), ds_scale, dq)
                else:
                    ds_scale *= 1.0 / (65504.0 if k_load.dtype == tl.float16 else 448.0)
                    ds_scale = tl.where(ds_scale <= 2e-38, 1.0, ds_scale)
                    ds = tl.mul(ds, tl.fdiv(1.0, ds_scale)).to(k_load.dtype)
                    dq = tl.fma(tl.dot(ds, k_load, out_dtype=tl.float32), ds_scale, dq)
            else:
                ds = ds.to(k_load.dtype)
                dq = tl.dot(ds, k_load, dq, out_dtype=tl.float32)

    dq = dq.to(dq_ptr.type.element_ty)
    dq_desc = tl.make_tensor_descriptor(dq_ptr + offset_q * QHD, shape=[QN, QHD], strides=[QHD, 1], block_shape=[BLOCK_SIZE_M, QHD])
    dq_desc.store([start_m_block, 0], dq)


@triton.autotune(
    configs=autotune_configs,
    key=[
        "is_causal", "do_mask",
        "QZ", "QH", "QN_AT", "QHD",
        "KZ", "KH", "KN_AT", "KHD",
        "VZ", "VH", "VN_AT", "VHD",
        "qk_is_quantized",
        "pv_is_quantized",
        "q_dtype", "v_dtype",
        "out_dtype", "return_dtype",
        "mask_dtype",
        "do_grad_k", "do_grad_v",
    ],
    cache_results=True,
)
@triton.jit
def sdnq_attn_bwd_dkv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr,
    q_scale_ptr, k_scale_ptr, v_scale_ptr, do_scale_ptr,
    dk_ptr, dv_ptr, lse_ptr, delta_ptr, mask_ptr,
    is_causal: tl.constexpr,
    do_mask: tl.constexpr,
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
    return_dtype: tl.constexpr, # pylint: disable=unused-argument
    mask_dtype: tl.constexpr, # pylint: disable=unused-argument
    do_grad_k: tl.constexpr,
    do_grad_v: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
) -> None:
    start_n = tl.program_id(0)
    off_h_k = tl.program_id(1)
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
    tl.assume(off_z >= 0)
    tl.assume(BLOCK_SIZE_M > 0)
    tl.assume(BLOCK_SIZE_N > 0)
    tl.assume(do_mask == 0 or do_mask == 1) # pylint: disable=consider-using-in
    tl.assume(is_causal == 0 or is_causal == 1) # pylint: disable=consider-using-in
    tl.assume(qk_is_quantized == 0 or qk_is_quantized == 1) # pylint: disable=consider-using-in
    tl.assume(pv_is_quantized == 0 or pv_is_quantized == 1) # pylint: disable=consider-using-in
    tl.assume(do_grad_k == 0 or do_grad_k == 1) # pylint: disable=consider-using-in
    tl.assume(do_grad_v == 0 or do_grad_v == 1) # pylint: disable=consider-using-in

    do_k_mask = KN % BLOCK_SIZE_N != 0
    start_n_block = start_n * BLOCK_SIZE_N
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n_block + tl.arange(0, BLOCK_SIZE_N)
    offset_k = off_z * (KN * KH) + off_h_k * KN
    offset_v = off_z * (VN * VH) + off_h_k * VN

    k_desc = tl.make_tensor_descriptor(k_ptr + offset_k * KHD, shape=[KN, KHD], strides=[KHD, 1], block_shape=[BLOCK_SIZE_N, KHD])
    v_desc = tl.make_tensor_descriptor(v_ptr + offset_v * VHD, shape=[VN, VHD], strides=[VHD, 1], block_shape=[BLOCK_SIZE_N, VHD])

    if qk_is_quantized:
        k_scale_desc = tl.make_tensor_descriptor(k_scale_ptr + offset_k, shape=[KN], strides=[1,], block_shape=[BLOCK_SIZE_N])
        k_scale = k_scale_desc.load([start_n_block])[None, :]
    if pv_is_quantized:
        v_scale_desc = tl.make_tensor_descriptor(v_scale_ptr + offset_v, shape=[VN], strides=[1,], block_shape=[BLOCK_SIZE_N])
        v_scale = v_scale_desc.load([start_n_block])[None, :]

    k = k_desc.load([start_n_block, 0]).T
    v = v_desc.load([start_n_block, 0]).T
    dk = tl.zeros([BLOCK_SIZE_N, KHD], dtype=tl.float32)
    dv = tl.zeros([BLOCK_SIZE_N, VHD], dtype=tl.float32)

    qh_ratio = QH // KH
    for qh_idx in tl.range(0, qh_ratio):
        off_h = off_h_k * qh_ratio + qh_idx
        offset_q = off_z * (QN * QH) + off_h * QN

        q_desc = tl.make_tensor_descriptor(q_ptr + offset_q * QHD, shape=[QN, QHD], strides=[QHD, 1], block_shape=[BLOCK_SIZE_M, QHD])
        do_desc = tl.make_tensor_descriptor(do_ptr + offset_q * QHD, shape=[QN, QHD], strides=[QHD, 1], block_shape=[BLOCK_SIZE_M, QHD])
        lse_desc = tl.make_tensor_descriptor(lse_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
        delta_desc = tl.make_tensor_descriptor(delta_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])

        if do_mask:
            mask_desc = tl.make_tensor_descriptor(mask_ptr + offset_q * MKN, shape=[MQN, MKN], strides=[MKN, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
        if qk_is_quantized:
            q_scale_desc = tl.make_tensor_descriptor(q_scale_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
        if pv_is_quantized:
            do_scale_desc = tl.make_tensor_descriptor(do_scale_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])

        if is_causal:
            start_m_idx_start = start_n_block // BLOCK_SIZE_M
        else:
            start_m_idx_start = 0

        for start_m_idx in tl.range(start_m_idx_start, tl.cdiv(QN, BLOCK_SIZE_M)):
            start_m = start_m_idx * BLOCK_SIZE_M
            skip = False
            if do_mask:
                mask = mask_desc.load([start_m, start_n_block])
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
                q = q_desc.load([start_m, 0])
                if qk_is_quantized:
                    q_scale = q_scale_desc.load([start_m])[:, None]
                    if q.dtype == tl.int8:
                        qk = tl.mul(tl.mul(tl.dot(q, k, out_dtype=tl.int32).to(tl.float32), q_scale), k_scale)
                    else:
                        qk = tl.mul(tl.mul(tl.dot(q, k, out_dtype=tl.float32), q_scale), k_scale)
                else:
                    qk = tl.dot(q, k, out_dtype=tl.float32)

                if is_causal and start_m < (start_n_block + BLOCK_SIZE_N):
                    qk = tl.where((start_m + offs_m[:, None]) >= offs_n[None, :], qk, float("-inf"))
                if do_mask:
                    if mask.dtype == tl.int1:
                        qk = tl.where(mask, qk, float("-inf"))
                    else:
                        qk += mask
                if do_k_mask and (start_n_block + BLOCK_SIZE_N) > KN:
                    qk = tl.where(offs_n[None, :] < KN, qk, float("-inf"))

                lse = lse_desc.load([start_m]).to(tl.float32)
                qk -= lse[:, None]
                p = tl.exp2(qk)

                do = do_desc.load([start_m, 0])
                if pv_is_quantized:
                    do_scale = do_scale_desc.load([start_m])[:, None]

                if do_grad_k:
                    if pv_is_quantized:
                        if do.dtype == tl.int8:
                            dp = tl.mul(tl.mul(tl.dot(do, v, out_dtype=tl.int32).to(tl.float32), do_scale), v_scale)
                        else:
                            dp = tl.mul(tl.mul(tl.dot(do, v, out_dtype=tl.float32), do_scale), v_scale)
                    else:
                        dp = tl.dot(do, v, out_dtype=tl.float32)

                    delta = delta_desc.load([start_m]).to(tl.float32)

                    ds = tl.mul(p, tl.mul(tl.sub(dp, delta[:, None]), 0.6931471805599453))
                    if qk_is_quantized:
                        ds *= q_scale
                        ds = ds.T
                        ds_scale = tl.max(tl.abs(ds), 1)[:, None]
                        if q.dtype == tl.int8:
                            ds_scale *= 1.0 / 127.0
                            ds_scale = tl.where(ds_scale <= 2e-38, 1.0, ds_scale)
                            ds = tl.floor(tl.fma(ds, (1.0 / ds_scale), 0.5)).to(tl.int8)
                            dk = tl.fma(tl.dot(ds, q, out_dtype=tl.int32).to(tl.float32), ds_scale, dk)
                        else:
                            ds_scale *= 1.0 / (65504.0 if q.dtype == tl.float16 else 448.0)
                            ds_scale = tl.where(ds_scale <= 2e-38, 1.0, ds_scale)
                            ds = tl.mul(ds, tl.fdiv(1.0, ds_scale)).to(q.dtype)
                            dk = tl.fma(tl.dot(ds, q, out_dtype=tl.float32), ds_scale, dk)
                    else:
                        ds = ds.T
                        ds = ds.to(q.dtype)
                        dk = tl.dot(ds, q, dk, out_dtype=tl.float32)

                if do_grad_v:
                    if pv_is_quantized:
                        p *= do_scale
                        p = p.T
                        p_scale = tl.max(p, 1)[:, None]
                        if do.dtype == tl.int8:
                            p_scale *= 1.0 / 127.0
                            p_scale = tl.where(p_scale <= 2e-38, 1.0, p_scale)
                            p = tl.floor(tl.fma(p, tl.fdiv(1.0, p_scale), 0.5)).to(tl.int8)
                            dv = tl.fma(tl.dot(p, do, out_dtype=tl.int32).to(tl.float32), p_scale, dv)
                        else:
                            p_scale *= 1.0 / (65504.0 if do.dtype == tl.float16 else 448.0)
                            p_scale = tl.where(p_scale <= 2e-38, 1.0, p_scale)
                            p = tl.mul(p, tl.fdiv(1.0, p_scale)).to(do.dtype)
                            dv = tl.fma(tl.dot(p, do, out_dtype=tl.float32), p_scale, dv)
                    else:
                        p = p.T
                        p = p.to(do.dtype)
                        dv = tl.dot(p, do, dv, out_dtype=tl.float32)

    if do_grad_k:
        dk = dk.to(dk_ptr.type.element_ty)
        dk_desc = tl.make_tensor_descriptor(dk_ptr + offset_k * KHD, shape=[KN, KHD], strides=[KHD, 1], block_shape=[BLOCK_SIZE_N, KHD])
        dk_desc.store([start_n_block, 0], dk)
    if do_grad_v:
        dv = dv.to(dv_ptr.type.element_ty)
        dv_desc = tl.make_tensor_descriptor(dv_ptr + offset_v * VHD, shape=[VN, VHD], strides=[VHD, 1], block_shape=[BLOCK_SIZE_N, VHD])
        dv_desc.store([start_n_block, 0], dv)


def get_attn_backward_inputs(grad_output: torch.FloatTensor, out: torch.FloatTensor, hadamard: torch.Tensor | None, pv_matmul_dtype: str | None = None) -> tuple[torch.FloatTensor, torch.FloatTensor | None, torch.FloatTensor, torch.FloatTensor]:
    out = out.contiguous()
    delta = torch.sum(torch.mul(out, grad_output), dim=-1, dtype=torch.float32)
    if pv_matmul_dtype in {"enabled", "uint8"}:
        pv_matmul_dtype = "int8"
    if pv_matmul_dtype not in {None, "auto", "none", "no", "disabled"}:
        if hadamard is not None:
            grad_output = apply_hadamard(grad_output, group_size=hadamard.shape[-1], hadamard=hadamard)[0]
        quantize_mm_func_pv = quantize_int_mm if pv_matmul_dtype.startswith("int") else quantize_fp_mm
        grad_output, grad_output_scale = quantize_mm_func_pv(grad_output.contiguous().to(dtype=torch.float32), dim=-1, matmul_dtype=pv_matmul_dtype)
        grad_output_scale = grad_output_scale.squeeze(-1)
    else:
        grad_output = grad_output.contiguous()
        grad_output_scale = None
    return grad_output, grad_output_scale, out, delta


def sdnq_triton_atten_bwd(
    grad_output: torch.FloatTensor,
    out: torch.FloatTensor,
    lse: torch.FloatTensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_scale: torch.FloatTensor | None,
    key_scale: torch.FloatTensor | None,
    value_scale: torch.FloatTensor | None,
    QHD: int,
    KHD: int,
    VHD: int,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    use_hadamard: bool = False,
    hadamard_group_size: int = 256,
    pv_matmul_dtype: str | None = None,
    do_quantize: bool = True,
    do_grad_q: bool = True,
    do_grad_k: bool = True,
    do_grad_v: bool = True,
) -> tuple[torch.FloatTensor | None, torch.FloatTensor | None, torch.FloatTensor | None]:
    if not do_grad_q and not do_grad_k and not do_grad_v:
        return None, None, None

    return_dtype = grad_output.dtype
    QZ, QH, QN, QHD = query.shape
    _, KH, KN, KHD = key.shape
    _, _, VN, VHD = value.shape

    if use_hadamard:
        hadamard = get_hadamard(hadamard_group_size, dtype=grad_output.dtype, device=grad_output.device)
    else:
        hadamard = None
    grad_output, grad_output_scale, out, delta = get_attn_backward_inputs(grad_output, out, hadamard=hadamard, pv_matmul_dtype=pv_matmul_dtype if do_quantize else "disabled")

    other_args = (
        (1 if is_causal else 0),
        (1 if attn_mask is not None else 0),
        *query.shape, *key.shape, *value.shape,
        *(attn_mask.shape if attn_mask is not None else (0, 0, 0, 0)),
        math.ceil(QN / min_block_size),
        math.ceil(KN / min_block_size),
        math.ceil(VN / min_block_size),
        (1 if query_scale is not None else 0),
        (1 if value_scale is not None else 0),
        str(query.dtype), str(value.dtype), str(out.dtype), str(return_dtype),
        str(attn_mask.dtype if attn_mask is not None else None),
    )

    if do_grad_q:
        def grid_dq(META):
            return (triton.cdiv(QN, META["BLOCK_SIZE_M"]), QH, QZ)
        dq = torch.empty_like(query, dtype=return_dtype)
        sdnq_attn_bwd_dq_kernel[grid_dq](
            query, key, value, grad_output,
            query_scale, key_scale, value_scale, grad_output_scale,
            dq, lse, delta, attn_mask, *other_args,
        )
        if use_hadamard:
            dq = rotate_hadamard_compiled(dq, group_size=hadamard_group_size, hadamard=hadamard)
        dq = dq[..., :QHD]
    else:
        dq = None

    if do_grad_k or do_grad_v:
        def grid_dkv(META):
            return (triton.cdiv(KN, META["BLOCK_SIZE_N"]), KH, QZ)
        dk = torch.empty_like(key, dtype=return_dtype) if do_grad_k else None
        dv = torch.empty_like(value, dtype=return_dtype) if do_grad_v else None
        sdnq_attn_bwd_dkv_kernel[grid_dkv](
            query, key, value, grad_output,
            query_scale, key_scale, value_scale, grad_output_scale,
            dk, dv, lse, delta, attn_mask, *other_args,
            (1 if do_grad_k else 0), (1 if do_grad_v else 0)
        )
        if do_grad_k:
            if use_hadamard:
                dk = rotate_hadamard_compiled(dk, group_size=hadamard_group_size, hadamard=hadamard)
            dk = dk[..., :KHD]
        if do_grad_v:
            if use_hadamard and pv_matmul_dtype not in {None, "auto", "none", "no", "disabled"}:
                dv = rotate_hadamard_compiled(dv, group_size=hadamard_group_size, hadamard=hadamard)
            dv = dv[..., :VHD]
    else:
        dk = None
        dv = None

    return dq, dk, dv, *(None,)*10


class SDNQAttenBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        scale: float | None,
        smooth_k: bool,
        use_hadamard: bool,
        hadamard_group_size: int,
        matmul_dtype: str,
        pv_matmul_dtype: str | None,
        do_quantize: bool,
        out_dtype: torch.dtype | None,
    ) -> torch.FloatTensor:
        ctx.QHD = query.shape[-1]
        ctx.KHD = key.shape[-1]
        ctx.VHD = value.shape[-1]
        ctx.is_causal = is_causal
        ctx.do_quantize = do_quantize
        ctx.pv_matmul_dtype = pv_matmul_dtype

        (
            out, lse, query, key, value,
            query_scale, key_scale, value_scale,
            attn_mask, use_hadamard, hadamard_group_size,
        ) = sdnq_triton_atten(
            query, key, value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=scale,
            smooth_k=smooth_k,
            use_hadamard=use_hadamard,
            hadamard_group_size=hadamard_group_size,
            matmul_dtype=matmul_dtype,
            pv_matmul_dtype=pv_matmul_dtype,
            do_quantize=do_quantize,
            out_dtype=out_dtype,
            return_backward=True,
        )

        ctx.use_hadamard = use_hadamard
        ctx.hadamard_group_size = hadamard_group_size
        ctx.save_for_backward(out, lse, query, key, value, query_scale, key_scale, value_scale, attn_mask)
        return out[..., :ctx.VHD]

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor) -> tuple[torch.FloatTensor | None, torch.FloatTensor | None, torch.FloatTensor | None]:
        out, lse, query, key, value, query_scale, key_scale, value_scale, attn_mask = ctx.saved_tensors
        return sdnq_triton_atten_bwd(
            grad_output, out, lse, query, key, value,
            query_scale, key_scale, value_scale,
            ctx.QHD, ctx.KHD, ctx.VHD,
            attn_mask=attn_mask,
            pv_matmul_dtype=ctx.pv_matmul_dtype,
            is_causal=ctx.is_causal,
            use_hadamard=ctx.use_hadamard,
            hadamard_group_size=ctx.hadamard_group_size,
            do_quantize=ctx.do_quantize,
            do_grad_q=ctx.needs_input_grad[0],
            do_grad_k=ctx.needs_input_grad[1],
            do_grad_v=ctx.needs_input_grad[2],
        )


def sdnq_triton_atten_with_backward(
        query: torch.FloatTensor,
        key: torch.FloatTensor,
        value: torch.FloatTensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0, # pylint: disable=unused-argument
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False, # pylint: disable=unused-argument
        smooth_k: bool = False,
        use_hadamard: bool = False,
        hadamard_group_size: int = 256,
        matmul_dtype: str = "int8",
        pv_matmul_dtype: str | None = None,
        do_quantize: bool = True,
        out_dtype: torch.dtype | None = None,
    ) -> torch.FloatTensor:
    return SDNQAttenBackward.apply(
        query, key, value,
        attn_mask,
        is_causal,
        scale,
        smooth_k,
        use_hadamard,
        hadamard_group_size,
        matmul_dtype,
        pv_matmul_dtype,
        do_quantize,
        out_dtype
    )


get_attn_backward_inputs = compile_func(get_attn_backward_inputs)
