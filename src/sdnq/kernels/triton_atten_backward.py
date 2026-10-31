import math

import torch
from torch.library import triton_op, wrap_triton

import triton
import triton.language as tl

from ..common import compile_func, inference_context
from ..quant_utils import quantize_int_mm, quantize_fp_mm, get_hadamard, rotate_hadamard, rotate_hadamard_compiled

from .triton_atten import (
    sdnq_triton_atten,
    autotune_configs,
    block_size_divisor,
    prune_configs,
    block_count_chunk,
    block_index_chunk,
    get_block_mask_input,
    check_block_lists,
    check_tensor_descriptors,
)


@triton.autotune(
    configs=autotune_configs,
    key=[
        "is_causal",
        "do_mask",
        "do_block_mask",
        "use_fp16_accum",
        "use_tensor_descriptors",
        "use_tensor_descriptors_for_scales",
        "use_tensor_descriptors_for_masks",
        "QZ", "QH", "QN_AT", "QHD",
        "KZ", "KH", "KN_AT", "KHD",
        "VZ", "VH", "VN_AT", "VHD",
        "block_size_divisor_at",
        "qk_is_quantized",
        "pv_is_quantized",
        "q_dtype", "v_dtype",
        "out_dtype", "mask_dtype",
        "BLOCK_MASK_M", "BLOCK_MASK_N",
    ],
    prune_configs_by={"early_config_prune": prune_configs},
    cache_results=True,
)
@triton.jit
def sdnq_attn_bwd_dq_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr,
    q_scale_ptr, k_scale_ptr, v_scale_ptr, do_scale_ptr,
    dq_ptr, lse_ptr, delta_ptr, mask_ptr, block_count_ptr, block_index_ptr, sm_scale,
    is_causal: tl.constexpr,
    do_mask: tl.constexpr,
    do_block_mask: tl.constexpr,
    use_fp16_accum: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    use_tensor_descriptors_for_scales: tl.constexpr,
    use_tensor_descriptors_for_masks: tl.constexpr,
    QZ: tl.constexpr, QH: tl.constexpr, QN: tl.constexpr, QHD: tl.constexpr,
    KZ: tl.constexpr, KH: tl.constexpr, KN: tl.constexpr, KHD: tl.constexpr,
    VZ: tl.constexpr, VH: tl.constexpr, VN: tl.constexpr, VHD: tl.constexpr,
    MZ: tl.constexpr, MH: tl.constexpr, MQN: tl.constexpr, MKN: tl.constexpr,
    BLOCK_MASK_M: tl.constexpr,
    BLOCK_MASK_N: tl.constexpr,
    BLOCK_COUNT_CHUNK: tl.constexpr,
    BLOCK_INDEX_CHUNK: tl.constexpr,
    BZ: tl.constexpr, BH: tl.constexpr,
    qk_is_quantized: tl.constexpr,
    pv_is_quantized: tl.constexpr,
    q_dtype: tl.constexpr, # pylint: disable=unused-argument
    v_dtype: tl.constexpr, # pylint: disable=unused-argument
    out_dtype: tl.constexpr, # pylint: disable=unused-argument
    mask_dtype: tl.constexpr, # pylint: disable=unused-argument
    QN_AT: tl.constexpr, # pylint: disable=unused-argument
    KN_AT: tl.constexpr, # pylint: disable=unused-argument
    VN_AT: tl.constexpr, # pylint: disable=unused-argument
    block_size_divisor_at: tl.constexpr, # pylint: disable=unused-argument
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
    tl.assume(BLOCK_MASK_M >= 0)
    tl.assume(BLOCK_MASK_N >= 0)
    tl.assume(BLOCK_COUNT_CHUNK > 0)
    tl.assume(BLOCK_INDEX_CHUNK > 0)
    tl.assume(BZ >= 0)
    tl.assume(BH >= 0)
    tl.assume(off_h >= 0)
    tl.assume(off_z >= 0)
    tl.assume(start_m >= 0)
    tl.assume(BLOCK_SIZE_M > 0)
    tl.assume(BLOCK_SIZE_N > 0)
    tl.assume(is_causal == 0 or is_causal == 1) # pylint: disable=consider-using-in
    tl.assume(do_mask == 0 or do_mask == 1) # pylint: disable=consider-using-in
    tl.assume(do_block_mask == 0 or do_block_mask == 1) # pylint: disable=consider-using-in
    tl.assume(use_fp16_accum == 0 or use_fp16_accum == 1) # pylint: disable=consider-using-in
    tl.assume(use_tensor_descriptors == 0 or use_tensor_descriptors == 1) # pylint: disable=consider-using-in
    tl.assume(use_tensor_descriptors_for_scales == 0 or use_tensor_descriptors_for_scales == 1) # pylint: disable=consider-using-in
    tl.assume(use_tensor_descriptors_for_masks == 0 or use_tensor_descriptors_for_masks == 1) # pylint: disable=consider-using-in
    tl.assume(qk_is_quantized == 0 or qk_is_quantized == 1) # pylint: disable=consider-using-in
    tl.assume(pv_is_quantized == 0 or pv_is_quantized == 1) # pylint: disable=consider-using-in

    sm_scale = sm_scale.to(tl.float32)
    inv_sm_scale = tl.fdiv(1.0, sm_scale)
    log2_sm_scale = sm_scale * 1.4426950408889634
    if do_block_mask:
        # a ragged last mask block puts whole sub tiles past KN, which the tail rule has to cover even when the tile divides KN
        do_k_mask = (KN % BLOCK_SIZE_N != 0) or (KN % BLOCK_MASK_N != 0)
    else:
        do_k_mask = KN % BLOCK_SIZE_N != 0
    start_m_block = start_m * BLOCK_SIZE_M
    offs_m = start_m_block + tl.arange(0, BLOCK_SIZE_M)
    blocks_n = tl.arange(0, BLOCK_SIZE_N)

    off_h_64 = off_h.to(tl.int64)
    off_z_64 = off_z.to(tl.int64)
    offset_q = off_z_64 * (QN * QH) + off_h_64 * QN
    offset_k = off_z_64 * (KN * KH) + ((off_h_64 * KH) // QH) * KN
    offset_v = off_z_64 * (VN * VH) + ((off_h_64 * VH) // QH) * VN

    if use_tensor_descriptors:
        q_desc = tl.make_tensor_descriptor(q_ptr + offset_q * QHD, shape=[QN, QHD], strides=[QHD, 1], block_shape=[BLOCK_SIZE_M, triton.next_power_of_2(QHD)])
        k_desc = tl.make_tensor_descriptor(k_ptr + offset_k * KHD, shape=[KN, KHD], strides=[KHD, 1], block_shape=[BLOCK_SIZE_N, triton.next_power_of_2(KHD)])
        v_desc = tl.make_tensor_descriptor(v_ptr + offset_v * VHD, shape=[VN, VHD], strides=[VHD, 1], block_shape=[BLOCK_SIZE_N, triton.next_power_of_2(VHD)])
        do_desc = tl.make_tensor_descriptor(do_ptr + offset_q * VHD, shape=[QN, VHD], strides=[VHD, 1], block_shape=[BLOCK_SIZE_M, triton.next_power_of_2(VHD)])
    else:
        offs_qhd = tl.arange(0, triton.next_power_of_2(QHD))
        offs_khd = tl.arange(0, triton.next_power_of_2(KHD))
        offs_vhd = tl.arange(0, triton.next_power_of_2(VHD))

    if use_tensor_descriptors_for_scales:
        lse_desc = tl.make_tensor_descriptor(lse_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
        delta_desc = tl.make_tensor_descriptor(delta_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
        if qk_is_quantized:
            q_scale_desc = tl.make_tensor_descriptor(q_scale_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
            k_scale_desc = tl.make_tensor_descriptor(k_scale_ptr + offset_k, shape=[KN], strides=[1,], block_shape=[BLOCK_SIZE_N])
            q_scale = q_scale_desc.load([start_m_block])[:, None].to(tl.float32)
        if pv_is_quantized:
            v_scale_desc = tl.make_tensor_descriptor(v_scale_ptr + offset_v, shape=[VN], strides=[1,], block_shape=[BLOCK_SIZE_N])
            do_scale_desc = tl.make_tensor_descriptor(do_scale_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
            do_scale = do_scale_desc.load([start_m_block])[:, None].to(tl.float32)
    else:
        if qk_is_quantized:
            q_scale = tl.load(q_scale_ptr + offset_q + offs_m[:, None], mask=offs_m[:, None] < QN, other=0.0).to(tl.float32)
        if pv_is_quantized:
            do_scale = tl.load(do_scale_ptr + offset_q + offs_m[:, None], mask=offs_m[:, None] < QN, other=0.0).to(tl.float32)

    if do_mask:
        if use_tensor_descriptors_for_masks:
            offset_mask = off_z_64 * ((MH * MQN) if (MZ != 1 and QZ != 1) else 0) + off_h_64 * (MQN if (MH != 1 and QH != 1) else 0)
            mask_desc = tl.make_tensor_descriptor(mask_ptr + offset_mask * MKN, shape=[QN, MKN], strides=[(MKN if (MQN != 1 and QN != 1) else 0), 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
            if MKN == 1 and KN != 1:
                mask_cols = tl.arange(0, BLOCK_SIZE_N)[None, :]
        else:
            mask_other = 0 if (mask_ptr.type.element_ty == tl.int1 or mask_ptr.type.element_ty == tl.int8) else 0.0 # pylint: disable=consider-using-in
            offset_mask = (off_z_64 * ((MH * MQN * MKN) if MZ != 1 else 0)) + (off_h_64 * ((MQN * MKN) if MH != 1 else 0)) + (offs_m[:, None] * (MKN if (MQN != 1 and QN != 1) else 0))

    if do_block_mask:
        # dq walks the same kv blocks the forward walked for this query block, so it reads the forward's lists unchanged
        tl.static_assert(BLOCK_MASK_M % BLOCK_SIZE_M == 0, "BLOCK_SIZE_M must divide BLOCK_MASK_M")
        tl.static_assert(BLOCK_MASK_N % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must divide BLOCK_MASK_N")
        SUB_TILES = BLOCK_MASK_N // BLOCK_SIZE_N
        NQB = (QN + BLOCK_MASK_M - 1) // BLOCK_MASK_M
        NKB = (KN + BLOCK_MASK_N - 1) // BLOCK_MASK_N
        NQB_PAD = ((NQB + BLOCK_COUNT_CHUNK - 1) // BLOCK_COUNT_CHUNK) * BLOCK_COUNT_CHUNK
        NKB_PAD = ((NKB + BLOCK_INDEX_CHUNK - 1) // BLOCK_INDEX_CHUNK) * BLOCK_INDEX_CHUNK
        q_block = start_m_block // BLOCK_MASK_M
        offs_count = tl.arange(0, BLOCK_COUNT_CHUNK)
        offs_index = tl.arange(0, BLOCK_INDEX_CHUNK)
        offset_lists = off_z_64 * (BH if BZ != 1 else 0) + (off_h_64 if BH != 1 else 0)
        if use_tensor_descriptors_for_scales:
            count_desc = tl.make_tensor_descriptor(block_count_ptr + offset_lists * NQB_PAD, shape=[NQB_PAD], strides=[1,], block_shape=[BLOCK_COUNT_CHUNK])
            counts = count_desc.load([(q_block // BLOCK_COUNT_CHUNK) * BLOCK_COUNT_CHUNK])
        else:
            index_base_ptr = block_index_ptr + (offset_lists * NQB + q_block) * NKB_PAD
            counts = tl.load(block_count_ptr + offset_lists * NQB_PAD + (q_block // BLOCK_COUNT_CHUNK) * BLOCK_COUNT_CHUNK + offs_count)
        num_tiles = tl.sum(tl.where(offs_count == (q_block % BLOCK_COUNT_CHUNK), counts, 0)) * SUB_TILES
        index_desc = tl.make_tensor_descriptor(block_index_ptr + (offset_lists * NQB + q_block) * NKB_PAD, shape=[NKB_PAD], strides=[1,], block_shape=[BLOCK_INDEX_CHUNK])
    else:
        num_tiles = tl.cdiv(KN, BLOCK_SIZE_N)

    if use_tensor_descriptors:
        q = q_desc.load([start_m_block, 0])
        do = do_desc.load([start_m_block, 0])
    else:
        q = tl.load(q_ptr + offset_q * QHD + offs_m[:, None] * QHD + offs_qhd[None, :], mask=(offs_m[:, None] < QN) & (offs_qhd[None, :] < QHD), other=0.0 if q_ptr.type.element_ty != tl.int8 else 0)
        do = tl.load(do_ptr + offset_q * VHD + offs_m[:, None] * VHD + offs_vhd[None, :], mask=(offs_m[:, None] < QN) & (offs_vhd[None, :] < VHD), other=0.0 if v_ptr.type.element_ty != tl.int8 else 0)

    if use_tensor_descriptors_for_scales:
        lse = lse_desc.load([start_m_block])[:, None].to(tl.float32)
        delta = delta_desc.load([start_m_block])[:, None].to(tl.float32)
    else:
        lse = tl.load(lse_ptr + offset_q + offs_m[:, None], mask=offs_m[:, None] < QN, other=0.0).to(tl.float32)
        delta = tl.load(delta_ptr + offset_q + offs_m[:, None], mask=offs_m[:, None] < QN, other=0.0).to(tl.float32)

    dq = tl.zeros([BLOCK_SIZE_M, triton.next_power_of_2(QHD)], dtype=tl.float32)

    if use_fp16_accum:
        if qk_is_quantized and q_ptr.type.element_ty == tl.float16:
            fp16_scale_qk = 65536.0 * KHD
            in_scale_qk = 1.0 / (65536.0 * KHD)**0.5
            q = tl.mul(q.to(tl.float32), in_scale_qk).to(tl.float16)
            fp16_scale_dq = 65536.0 * BLOCK_SIZE_N
            in_scale_dq = 1.0 / (65536.0 * BLOCK_SIZE_N)**0.5
        if pv_is_quantized and v_ptr.type.element_ty == tl.float16:
            fp16_scale_pv = 65536.0 * VHD
            in_scale_pv = 1.0 / (65536.0 * VHD)**0.5
            do = tl.mul(do.to(tl.float32), in_scale_pv).to(tl.float16)
            do_scale = tl.mul(do_scale, fp16_scale_pv)

    for tile in tl.range(0, num_tiles):
        if do_block_mask:
            block_pos = tile // SUB_TILES
            if use_tensor_descriptors_for_scales:
                chunk = index_desc.load([(block_pos // BLOCK_INDEX_CHUNK) * BLOCK_INDEX_CHUNK])
            else:
                chunk = tl.load(index_base_ptr + offs_index + (block_pos // BLOCK_INDEX_CHUNK) * BLOCK_INDEX_CHUNK)
            block_n = tl.sum(tl.where(offs_index == (block_pos % BLOCK_INDEX_CHUNK), chunk, 0))
            start_n = block_n * BLOCK_MASK_N + (tile % SUB_TILES) * BLOCK_SIZE_N
        else:
            start_n = tile * BLOCK_SIZE_N
        offs_n = start_n + blocks_n
        if not (is_causal and ((start_m_block + BLOCK_SIZE_M) <= start_n)):
            skip = False
            if do_mask:
                if use_tensor_descriptors_for_masks:
                    if MKN == 1 and KN != 1:
                        mask = mask_desc.load([start_m_block, 0])
                        if mask.dtype == tl.int8 or mask.dtype == tl.int1: # pylint: disable=consider-using-in
                            mask = tl.max(tl.where(mask_cols == 0, mask, 0), axis=1, keep_dims=True).to(tl.int1)
                        else:
                            mask = tl.max(tl.where(mask_cols == 0, mask, float("-inf")), axis=1, keep_dims=True).to(tl.float32)
                    else:
                        mask = mask_desc.load([start_m_block, start_n])
                else:
                    if MKN == 1 and KN != 1:
                        mask = tl.load(mask_ptr + offset_mask, mask=offs_m[:, None] < QN, other=mask_other)
                    else:
                        mask = tl.load(mask_ptr + offset_mask + offs_n[None, :], mask=(offs_m[:, None] < QN) & (offs_n[None, :] < MKN), other=mask_other)
                mask_max = tl.max(mask)
                if mask.dtype == tl.int8:
                    mask = mask.to(tl.int1)
                elif mask.dtype != tl.int1:
                    mask = mask.to(tl.float32)
                if mask.dtype == tl.int1:
                    if mask_max == 0:
                        skip = True
                elif mask_max == float("-inf"):
                    skip = True
            if not skip:
                if use_tensor_descriptors:
                    k = k_desc.load([start_n, 0])
                else:
                    k = tl.load(k_ptr + offset_k * KHD + offs_n[:, None] * KHD + offs_khd[None, :], mask=(offs_n[:, None] < KN) & (offs_khd[None, :] < KHD), other=0.0 if k_ptr.type.element_ty != tl.int8 else 0)
                if qk_is_quantized:
                    if use_tensor_descriptors_for_scales:
                        k_scale = k_scale_desc.load([start_n])[None, :].to(tl.float32)
                    else:
                        k_scale = tl.load(k_scale_ptr + offset_k + offs_n[None, :], mask=offs_n[None, :] < KN, other=0.0).to(tl.float32)
                    if q.dtype == tl.int8:
                        qk = tl.mul(tl.mul(tl.dot(q, k.T, out_dtype=tl.int32).to(tl.float32), q_scale), k_scale) #pylint: disable=used-before-assignment,possibly-used-before-assignment
                    elif use_fp16_accum and q.dtype == tl.float16:
                        k_T = tl.mul(k.T.to(tl.float32), in_scale_qk).to(tl.float16)
                        qk = tl.mul(tl.mul(tl.mul(tl.dot(q, k_T, out_dtype=tl.float16).to(tl.float32), q_scale), k_scale), fp16_scale_qk) #pylint: disable=used-before-assignment,possibly-used-before-assignment
                    else:
                        qk = tl.mul(tl.mul(tl.dot(q, k.T, out_dtype=tl.float32), q_scale), k_scale)
                else:
                    qk = tl.dot(q, k.T, out_dtype=tl.float32)

                if is_causal and start_m_block < (start_n + BLOCK_SIZE_N):
                    qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
                if do_mask:
                    if mask.dtype == tl.int1:
                        qk = tl.where(mask, qk, float("-inf"))
                    else:
                        qk = tl.fma(mask, inv_sm_scale, qk)
                if do_k_mask and (start_n + BLOCK_SIZE_N) > KN:
                    qk = tl.where(offs_n[None, :] < KN, qk, float("-inf"))

                qk = tl.fma(qk, log2_sm_scale, lse)
                p = tl.exp2(qk)

                if use_tensor_descriptors:
                    v = v_desc.load([start_n, 0]).T
                else:
                    v = tl.load(v_ptr + offset_v * VHD + offs_n[None, :] * VHD + offs_vhd[:, None], mask=(offs_n[None, :] < VN) & (offs_vhd[:, None] < VHD), other=0.0 if v_ptr.type.element_ty != tl.int8 else 0)
                if pv_is_quantized:
                    if use_tensor_descriptors_for_scales:
                        v_scale = v_scale_desc.load([start_n])[None, :].to(tl.float32)
                    else:
                        v_scale = tl.load(v_scale_ptr + offset_v + offs_n[None, :], mask=offs_n[None, :] < VN, other=0.0).to(tl.float32)
                    if do.dtype == tl.int8:
                        dp = tl.mul(tl.mul(tl.dot(do, v, out_dtype=tl.int32).to(tl.float32), do_scale), v_scale)
                    else:
                        if use_fp16_accum and v.dtype == tl.float16:
                            v = tl.mul(v.to(tl.float32), in_scale_pv).to(tl.float16)
                            dp = tl.mul(tl.mul(tl.dot(do, v, out_dtype=tl.float16).to(tl.float32), do_scale), v_scale)
                        else:
                            dp = tl.mul(tl.mul(tl.dot(do, v, out_dtype=tl.float32), do_scale), v_scale)
                else:
                    dp = tl.dot(do, v, out_dtype=tl.float32)

                ds = tl.mul(p, tl.sub(dp, delta))
                if qk_is_quantized:
                    ds *= k_scale
                    ds_scale = tl.max(tl.abs(ds), 1)[:, None]
                    if k.dtype == tl.int8:
                        ds_scale *= 1.0 / 127.0
                        ds_scale = tl.where(ds_scale <= 2e-38, 1.0, ds_scale)
                        ds = tl.floor(tl.fma(ds, (1.0 / ds_scale), 0.5)).to(tl.int8)
                        dq = tl.fma(tl.dot(ds, k, out_dtype=tl.int32).to(tl.float32), ds_scale, dq)
                    else:
                        ds_scale *= 1.0 / (65504.0 if k.dtype == tl.float16 else 448.0)
                        ds_scale = tl.where(ds_scale <= 2e-38, 1.0, ds_scale)
                        ds = tl.mul(ds, tl.fdiv(1.0, ds_scale)).to(k.dtype)
                        if use_fp16_accum and k.dtype == tl.float16:
                            ds_scale *= fp16_scale_dq # pylint: disable=used-before-assignment,possibly-used-before-assignment
                            ds = tl.mul(ds.to(tl.float32), in_scale_dq).to(tl.float16) # pylint: disable=used-before-assignment,possibly-used-before-assignment
                            k = tl.mul(k.to(tl.float32), in_scale_dq).to(tl.float16) # pylint: disable=used-before-assignment,possibly-used-before-assignment
                            dq = tl.fma(tl.dot(ds, k, out_dtype=tl.float16).to(tl.float32), ds_scale, dq)
                        else:
                            dq = tl.fma(tl.dot(ds, k, out_dtype=tl.float32), ds_scale, dq)
                else:
                    ds = ds.to(k.dtype)
                    dq = tl.dot(ds, k, dq, out_dtype=tl.float32)

    dq *= sm_scale
    dq = dq.to(dq_ptr.type.element_ty)
    if use_tensor_descriptors:
        dq_desc = tl.make_tensor_descriptor(dq_ptr + offset_q * QHD, shape=[QN, QHD], strides=[QHD, 1], block_shape=[BLOCK_SIZE_M, triton.next_power_of_2(QHD)])
        dq_desc.store([start_m_block, 0], dq)
    else:
        tl.store(dq_ptr + offset_q * QHD + offs_m[:, None] * QHD + offs_qhd[None, :], dq, mask=(offs_m[:, None] < QN) & (offs_qhd[None, :] < QHD))


@triton.autotune(
    configs=autotune_configs,
    key=[
        "do_grad_k",
        "do_grad_v",
        "is_causal",
        "do_mask",
        "do_block_mask",
        "use_fp16_accum",
        "use_tensor_descriptors",
        "use_tensor_descriptors_for_scales",
        "use_tensor_descriptors_for_masks",
        "QZ", "QH", "QN_AT", "QHD",
        "KZ", "KH", "KN_AT", "KHD",
        "VZ", "VH", "VN_AT", "VHD",
        "block_size_divisor_at",
        "qk_is_quantized",
        "pv_is_quantized",
        "q_dtype", "v_dtype",
        "out_dtype", "mask_dtype",
        "BLOCK_MASK_M", "BLOCK_MASK_N",
    ],
    prune_configs_by={"early_config_prune": prune_configs},
    cache_results=True,
)
@triton.jit
def sdnq_attn_bwd_dkv_kernel(
    q_ptr, k_ptr, v_ptr, do_ptr,
    q_scale_ptr, k_scale_ptr, v_scale_ptr, do_scale_ptr,
    dk_ptr, dv_ptr, lse_ptr, delta_ptr, mask_ptr, block_count_ptr, block_index_ptr, sm_scale,
    do_grad_k: tl.constexpr,
    do_grad_v: tl.constexpr,
    is_causal: tl.constexpr,
    do_mask: tl.constexpr,
    do_block_mask: tl.constexpr,
    use_fp16_accum: tl.constexpr,
    use_tensor_descriptors: tl.constexpr,
    use_tensor_descriptors_for_scales: tl.constexpr,
    use_tensor_descriptors_for_masks: tl.constexpr,
    QZ: tl.constexpr, QH: tl.constexpr, QN: tl.constexpr, QHD: tl.constexpr,
    KZ: tl.constexpr, KH: tl.constexpr, KN: tl.constexpr, KHD: tl.constexpr,
    VZ: tl.constexpr, VH: tl.constexpr, VN: tl.constexpr, VHD: tl.constexpr,
    MZ: tl.constexpr, MH: tl.constexpr, MQN: tl.constexpr, MKN: tl.constexpr,
    BLOCK_MASK_M: tl.constexpr,
    BLOCK_MASK_N: tl.constexpr,
    BLOCK_COUNT_CHUNK: tl.constexpr,
    BLOCK_INDEX_CHUNK: tl.constexpr,
    BZ: tl.constexpr, BH: tl.constexpr,
    qk_is_quantized: tl.constexpr,
    pv_is_quantized: tl.constexpr,
    q_dtype: tl.constexpr, # pylint: disable=unused-argument
    v_dtype: tl.constexpr, # pylint: disable=unused-argument
    out_dtype: tl.constexpr, # pylint: disable=unused-argument
    mask_dtype: tl.constexpr, # pylint: disable=unused-argument
    QN_AT: tl.constexpr, # pylint: disable=unused-argument
    KN_AT: tl.constexpr, # pylint: disable=unused-argument
    VN_AT: tl.constexpr, # pylint: disable=unused-argument
    block_size_divisor_at: tl.constexpr, # pylint: disable=unused-argument
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
    tl.assume(BLOCK_MASK_M >= 0)
    tl.assume(BLOCK_MASK_N >= 0)
    tl.assume(BLOCK_COUNT_CHUNK > 0)
    tl.assume(BLOCK_INDEX_CHUNK > 0)
    tl.assume(BZ >= 0)
    tl.assume(BH >= 0)
    tl.assume(off_z >= 0)
    tl.assume(BLOCK_SIZE_M > 0)
    tl.assume(BLOCK_SIZE_N > 0)
    tl.assume(is_causal == 0 or is_causal == 1) # pylint: disable=consider-using-in
    tl.assume(do_mask == 0 or do_mask == 1) # pylint: disable=consider-using-in
    tl.assume(do_block_mask == 0 or do_block_mask == 1) # pylint: disable=consider-using-in
    tl.assume(use_fp16_accum == 0 or use_fp16_accum == 1) # pylint: disable=consider-using-in
    tl.assume(use_tensor_descriptors == 0 or use_tensor_descriptors == 1) # pylint: disable=consider-using-in
    tl.assume(use_tensor_descriptors_for_scales == 0 or use_tensor_descriptors_for_scales == 1) # pylint: disable=consider-using-in
    tl.assume(use_tensor_descriptors_for_masks == 0 or use_tensor_descriptors_for_masks == 1) # pylint: disable=consider-using-in
    tl.assume(qk_is_quantized == 0 or qk_is_quantized == 1) # pylint: disable=consider-using-in
    tl.assume(pv_is_quantized == 0 or pv_is_quantized == 1) # pylint: disable=consider-using-in
    tl.assume(do_grad_k == 0 or do_grad_k == 1) # pylint: disable=consider-using-in
    tl.assume(do_grad_v == 0 or do_grad_v == 1) # pylint: disable=consider-using-in

    sm_scale = sm_scale.to(tl.float32)
    inv_sm_scale = tl.fdiv(1.0, sm_scale)
    log2_sm_scale = sm_scale * 1.4426950408889634
    do_k_mask = KN % BLOCK_SIZE_N != 0
    start_n_block = start_n * BLOCK_SIZE_N
    blocks_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = start_n_block + tl.arange(0, BLOCK_SIZE_N)

    off_h_k_64 = off_h_k.to(tl.int64)
    off_z_64 = off_z.to(tl.int64)
    offset_k = off_z_64 * (KN * KH) + off_h_k_64 * KN
    offset_v = off_z_64 * (VN * VH) + off_h_k_64 * VN

    if use_tensor_descriptors:
        k_desc = tl.make_tensor_descriptor(k_ptr + offset_k * KHD, shape=[KN, KHD], strides=[KHD, 1], block_shape=[BLOCK_SIZE_N, triton.next_power_of_2(KHD)])
        v_desc = tl.make_tensor_descriptor(v_ptr + offset_v * VHD, shape=[VN, VHD], strides=[VHD, 1], block_shape=[BLOCK_SIZE_N, triton.next_power_of_2(VHD)])
    else:
        offs_qhd = tl.arange(0, triton.next_power_of_2(QHD))
        offs_khd = tl.arange(0, triton.next_power_of_2(KHD))
        offs_vhd = tl.arange(0, triton.next_power_of_2(VHD))

    if use_tensor_descriptors_for_scales:
        if qk_is_quantized:
            k_scale_desc = tl.make_tensor_descriptor(k_scale_ptr + offset_k, shape=[KN], strides=[1,], block_shape=[BLOCK_SIZE_N])
            k_scale = k_scale_desc.load([start_n_block])[None, :].to(tl.float32)
        if pv_is_quantized:
            v_scale_desc = tl.make_tensor_descriptor(v_scale_ptr + offset_v, shape=[VN], strides=[1,], block_shape=[BLOCK_SIZE_N])
            v_scale = v_scale_desc.load([start_n_block])[None, :].to(tl.float32)
    else:
        if qk_is_quantized:
            k_scale = tl.load(k_scale_ptr + offset_k + offs_n[None, :], mask=offs_n[None, :] < KN, other=0.0).to(tl.float32)
        if pv_is_quantized:
            v_scale = tl.load(v_scale_ptr + offset_v + offs_n[None, :], mask=offs_n[None, :] < VN, other=0.0).to(tl.float32)

    if use_tensor_descriptors:
        k = k_desc.load([start_n_block, 0]).T
        v = v_desc.load([start_n_block, 0]).T
    else:
        k = tl.load(k_ptr + offset_k * KHD + offs_n[None, :] * KHD + offs_khd[:, None], mask=(offs_n[None, :] < KN) & (offs_khd[:, None] < KHD), other=0.0 if k_ptr.type.element_ty != tl.int8 else 0)
        v = tl.load(v_ptr + offset_v * VHD + offs_n[None, :] * VHD + offs_vhd[:, None], mask=(offs_n[None, :] < VN) & (offs_vhd[:, None] < VHD), other=0.0 if v_ptr.type.element_ty != tl.int8 else 0)

    if do_grad_k:
        dk_t = tl.zeros([triton.next_power_of_2(KHD), BLOCK_SIZE_N], dtype=tl.float32)
    if do_grad_v:
        dv_t = tl.zeros([triton.next_power_of_2(VHD), BLOCK_SIZE_N], dtype=tl.float32)

    if use_fp16_accum:
        if qk_is_quantized and q_ptr.type.element_ty == tl.float16:
            fp16_scale_qk = 65536.0 * KHD
            in_scale_qk = 1.0 / (65536.0 * KHD)**0.5
            k = tl.mul(k.to(tl.float32), in_scale_qk).to(tl.float16)
            fp16_scale_dk = 65536.0 * BLOCK_SIZE_M
            in_scale_dk = 1.0 / (65536.0 * BLOCK_SIZE_M)**0.5
        if pv_is_quantized and v_ptr.type.element_ty == tl.float16:
            fp16_scale_pv = 65536.0 * VHD
            in_scale_pv = 1.0 / (65536.0 * VHD)**0.5
            v = tl.mul(v.to(tl.float32), in_scale_pv).to(tl.float16)
            v_scale = tl.mul(v_scale, fp16_scale_pv)
            fp16_scale_dv = 65536.0 * BLOCK_SIZE_M
            in_scale_dv = 1.0 / (65536.0 * BLOCK_SIZE_M)**0.5

    qh_ratio = QH // KH
    for qh_idx in tl.range(0, qh_ratio):
        off_h_64 = off_h_k * qh_ratio + qh_idx
        offset_q = off_z_64 * (QN * QH) + off_h_64 * QN

        if use_tensor_descriptors:
            q_desc = tl.make_tensor_descriptor(q_ptr + offset_q * QHD, shape=[QN, QHD], strides=[QHD, 1], block_shape=[BLOCK_SIZE_M, triton.next_power_of_2(QHD)])
            do_desc = tl.make_tensor_descriptor(do_ptr + offset_q * VHD, shape=[QN, VHD], strides=[VHD, 1], block_shape=[BLOCK_SIZE_M, triton.next_power_of_2(VHD)])

        if use_tensor_descriptors_for_scales:
            lse_desc = tl.make_tensor_descriptor(lse_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
            delta_desc = tl.make_tensor_descriptor(delta_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
            if qk_is_quantized:
                q_scale_desc = tl.make_tensor_descriptor(q_scale_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])
            if pv_is_quantized:
                do_scale_desc = tl.make_tensor_descriptor(do_scale_ptr + offset_q, shape=[QN], strides=[1,], block_shape=[BLOCK_SIZE_M])

        if do_mask:
            if use_tensor_descriptors_for_masks:
                offset_mask = off_z_64 * ((MH * MQN) if (MZ != 1 and QZ != 1) else 0) + off_h_64 * (MQN if (MH != 1 and QH != 1) else 0)
                mask_desc = tl.make_tensor_descriptor(mask_ptr + offset_mask * MKN, shape=[QN, MKN], strides=[(MKN if (MQN != 1 and QN != 1) else 0), 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
                if MKN == 1 and KN != 1:
                    mask_cols = tl.arange(0, BLOCK_SIZE_N)[None, :]
            else:
                stride_mqn = (MKN if (MQN != 1 and QN != 1) else 0)
                mask_other = 0 if (mask_ptr.type.element_ty == tl.int1 or mask_ptr.type.element_ty == tl.int8) else 0.0 # pylint: disable=consider-using-in
                offset_mask_base = (off_z_64 * ((MH * MQN * MKN) if MZ != 1 else 0)) + (off_h_64 * ((MQN * MKN) if MH != 1 else 0))

        if is_causal:
            start_m_idx_start = start_n_block // BLOCK_SIZE_M
        else:
            start_m_idx_start = 0

        if do_block_mask:
            # dkv walks the query blocks that kept this kv block, so it reads the mask transposed: a count and an
            # ascending index list per kv block. A query tile past a ragged last block loads zeros for q and do,
            # which contribute nothing to either gradient, and a causally dead tile is scored to -inf as before
            tl.static_assert(BLOCK_MASK_M % BLOCK_SIZE_M == 0, "BLOCK_SIZE_M must divide BLOCK_MASK_M")
            tl.static_assert(BLOCK_MASK_N % BLOCK_SIZE_N == 0, "BLOCK_SIZE_N must divide BLOCK_MASK_N")
            SUB_TILES = BLOCK_MASK_M // BLOCK_SIZE_M
            NQB = (QN + BLOCK_MASK_M - 1) // BLOCK_MASK_M
            NKB = (KN + BLOCK_MASK_N - 1) // BLOCK_MASK_N
            NKB_PAD = ((NKB + BLOCK_COUNT_CHUNK - 1) // BLOCK_COUNT_CHUNK) * BLOCK_COUNT_CHUNK
            NQB_PAD = ((NQB + BLOCK_INDEX_CHUNK - 1) // BLOCK_INDEX_CHUNK) * BLOCK_INDEX_CHUNK
            kv_block = start_n_block // BLOCK_MASK_N
            offs_count = tl.arange(0, BLOCK_COUNT_CHUNK)
            offs_index = tl.arange(0, BLOCK_INDEX_CHUNK)
            offset_lists = off_z_64 * (BH if BZ != 1 else 0) + (off_h_64 if BH != 1 else 0)
            if use_tensor_descriptors_for_scales:
                index_desc = tl.make_tensor_descriptor(block_index_ptr + (offset_lists * NKB + kv_block) * NQB_PAD, shape=[NQB_PAD], strides=[1,], block_shape=[BLOCK_INDEX_CHUNK])
                count_desc = tl.make_tensor_descriptor(block_count_ptr + offset_lists * NKB_PAD, shape=[NKB_PAD], strides=[1,], block_shape=[BLOCK_COUNT_CHUNK])
                counts = count_desc.load([(kv_block // BLOCK_COUNT_CHUNK) * BLOCK_COUNT_CHUNK])
            else:
                index_base_ptr = block_index_ptr + (offset_lists * NKB + kv_block) * NQB_PAD
                counts = tl.load(block_count_ptr + offset_lists * NKB_PAD + (kv_block // BLOCK_COUNT_CHUNK) * BLOCK_COUNT_CHUNK + offs_count)
            num_tiles = tl.sum(tl.where(offs_count == (kv_block % BLOCK_COUNT_CHUNK), counts, 0)) * SUB_TILES
        else:
            num_tiles = tl.cdiv(QN, BLOCK_SIZE_M) - start_m_idx_start

        for tile in tl.range(0, num_tiles):
            if do_block_mask:
                block_pos = tile // SUB_TILES
                if use_tensor_descriptors_for_scales:
                    chunk = index_desc.load([(block_pos // BLOCK_INDEX_CHUNK) * BLOCK_INDEX_CHUNK])
                else:
                    chunk = tl.load(index_base_ptr + offs_index + (block_pos // BLOCK_INDEX_CHUNK) * BLOCK_INDEX_CHUNK)
                block_m = tl.sum(tl.where(offs_index == (block_pos % BLOCK_INDEX_CHUNK), chunk, 0))
                start_m = block_m * BLOCK_MASK_M + (tile % SUB_TILES) * BLOCK_SIZE_M
            else:
                start_m = (start_m_idx_start + tile) * BLOCK_SIZE_M
            offs_m = start_m + blocks_m
            skip = False
            if do_mask:
                if use_tensor_descriptors_for_masks:
                    if MKN == 1 and KN != 1:
                        mask = mask_desc.load([start_m, 0])
                        if mask.dtype == tl.int8 or mask.dtype == tl.int1: # pylint: disable=consider-using-in
                            mask = tl.max(tl.where(mask_cols == 0, mask, 0), axis=1, keep_dims=True).to(tl.int1)
                        else:
                            mask = tl.max(tl.where(mask_cols == 0, mask, float("-inf")), axis=1, keep_dims=True).to(tl.float32)
                    else:
                        mask = mask_desc.load([start_m, start_n_block])
                else:
                    offset_mask = offset_mask_base + (offs_m[:, None] * stride_mqn)
                    if MKN == 1 and KN != 1:
                        mask = tl.load(mask_ptr + offset_mask, mask=offs_m[:, None] < QN, other=mask_other)
                    else:
                        mask = tl.load(mask_ptr + offset_mask + offs_n[None, :], mask=(offs_m[:, None] < QN) & (offs_n[None, :] < MKN), other=mask_other)
                mask_max = tl.max(mask)
                if mask.dtype == tl.int8:
                    mask = mask.to(tl.int1)
                elif mask.dtype != tl.int1:
                    mask = mask.to(tl.float32)
                if mask.dtype == tl.int1:
                    if mask_max == 0:
                        skip = True
                elif mask_max == float("-inf"):
                    skip = True
            if not skip:
                if use_tensor_descriptors:
                    q = q_desc.load([start_m, 0])
                else:
                    q = tl.load(q_ptr + offset_q * QHD + offs_m[:, None] * QHD + offs_qhd[None, :], mask=(offs_m[:, None] < QN) & (offs_qhd[None, :] < QHD), other=0.0 if q_ptr.type.element_ty != tl.int8 else 0)
                if qk_is_quantized:
                    if use_tensor_descriptors_for_scales:
                        q_scale = q_scale_desc.load([start_m])[:, None].to(tl.float32)
                    else:
                        q_scale = tl.load(q_scale_ptr + offset_q + offs_m[:, None], mask=offs_m[:, None] < QN, other=0.0).to(tl.float32)
                    if q.dtype == tl.int8:
                        qk = tl.mul(tl.mul(tl.dot(q, k, out_dtype=tl.int32).to(tl.float32), q_scale), k_scale) #pylint: disable=used-before-assignment,possibly-used-before-assignment
                    elif use_fp16_accum and q.dtype == tl.float16:
                        q_k = tl.mul(q.to(tl.float32), in_scale_qk).to(tl.float16)
                        qk = tl.mul(tl.mul(tl.mul(tl.dot(q_k, k, out_dtype=tl.float16).to(tl.float32), q_scale), k_scale), fp16_scale_qk) #pylint: disable=used-before-assignment,possibly-used-before-assignment
                    else:
                        qk = tl.mul(tl.mul(tl.dot(q, k, out_dtype=tl.float32), q_scale), k_scale)
                else:
                    qk = tl.dot(q, k, out_dtype=tl.float32)

                if is_causal and start_m < (start_n_block + BLOCK_SIZE_N):
                    qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
                if do_mask:
                    if mask.dtype == tl.int1:
                        qk = tl.where(mask, qk, float("-inf"))
                    else:
                        qk = tl.fma(mask, inv_sm_scale, qk)
                if do_k_mask and (start_n_block + BLOCK_SIZE_N) > KN:
                    qk = tl.where(offs_n[None, :] < KN, qk, float("-inf"))

                if use_tensor_descriptors_for_scales:
                    lse = lse_desc.load([start_m])[:, None].to(tl.float32)
                else:
                    lse = tl.load(lse_ptr + offset_q + offs_m[:, None], mask=offs_m[:, None] < QN, other=0.0).to(tl.float32)

                qk = tl.fma(qk, log2_sm_scale, lse)
                p = tl.exp2(qk)

                if use_tensor_descriptors:
                    do = do_desc.load([start_m, 0])
                else:
                    do = tl.load(do_ptr + offset_q * VHD + offs_m[:, None] * VHD + offs_vhd[None, :], mask=(offs_m[:, None] < QN) & (offs_vhd[None, :] < VHD), other=0.0 if v_ptr.type.element_ty != tl.int8 else 0)

                if pv_is_quantized:
                    if use_tensor_descriptors_for_scales:
                        do_scale = do_scale_desc.load([start_m])[:, None].to(tl.float32)
                    else:
                        do_scale = tl.load(do_scale_ptr + offset_q + offs_m[:, None], mask=offs_m[:, None] < QN, other=0.0).to(tl.float32)

                if do_grad_k:
                    if pv_is_quantized:
                        if do.dtype == tl.int8:
                            dp = tl.mul(tl.mul(tl.dot(do, v, out_dtype=tl.int32).to(tl.float32), do_scale), v_scale)
                        if use_fp16_accum and v.dtype == tl.float16:
                            do_pv = tl.mul(do.to(tl.float32), in_scale_pv).to(tl.float16)
                            dp = tl.mul(tl.mul(tl.dot(do_pv, v, out_dtype=tl.float16).to(tl.float32), do_scale), v_scale)
                        else:
                            dp = tl.mul(tl.mul(tl.dot(do, v, out_dtype=tl.float32), do_scale), v_scale)
                    else:
                        dp = tl.dot(do, v, out_dtype=tl.float32)

                    if use_tensor_descriptors_for_scales:
                        delta = delta_desc.load([start_m])[:, None].to(tl.float32)
                    else:
                        delta = tl.load(delta_ptr + offset_q + offs_m[:, None], mask=offs_m[:, None] < QN, other=0.0).to(tl.float32)

                    ds = tl.mul(p, tl.sub(dp, delta))
                    if qk_is_quantized:
                        ds *= q_scale
                        ds_scale = tl.max(tl.abs(ds), 0)[None, :]
                        if q.dtype == tl.int8:
                            ds_scale *= 1.0 / 127.0
                            ds_scale = tl.where(ds_scale <= 2e-38, 1.0, ds_scale)
                            ds = tl.floor(tl.fma(ds, (1.0 / ds_scale), 0.5)).to(tl.int8)
                            dk_t = tl.fma(tl.dot(q.T, ds, out_dtype=tl.int32).to(tl.float32), ds_scale, dk_t)
                        else:
                            ds_scale *= 1.0 / (65504.0 if q.dtype == tl.float16 else 448.0)
                            ds_scale = tl.where(ds_scale <= 2e-38, 1.0, ds_scale)
                            ds = tl.mul(ds, tl.fdiv(1.0, ds_scale)).to(q.dtype)
                            if use_fp16_accum and q.dtype == tl.float16:
                                ds_scale *= fp16_scale_dk # pylint: disable=used-before-assignment,possibly-used-before-assignment
                                q_T = tl.mul(q.T.to(tl.float32), in_scale_dk).to(tl.float16) # pylint: disable=used-before-assignment,possibly-used-before-assignment
                                ds = tl.mul(ds.to(tl.float32), in_scale_dk).to(tl.float16) # pylint: disable=used-before-assignment,possibly-used-before-assignment
                                dk_t = tl.fma(tl.dot(q_T, ds, out_dtype=tl.float16).to(tl.float32), ds_scale, dk_t)
                            else:
                                dk_t = tl.fma(tl.dot(q.T, ds, out_dtype=tl.float32), ds_scale, dk_t)
                    else:
                        ds = ds.to(q.dtype)
                        dk_t = tl.dot(q.T, ds, dk_t, out_dtype=tl.float32)

                if do_grad_v:
                    if pv_is_quantized:
                        p *= do_scale
                        p_scale = tl.max(p, 0)[None, :]
                        if do.dtype == tl.int8:
                            p_scale *= 1.0 / 127.0
                            p_scale = tl.where(p_scale <= 2e-38, 1.0, p_scale)
                            p = tl.floor(tl.fma(p, tl.fdiv(1.0, p_scale), 0.5)).to(tl.int8)
                            dv_t = tl.fma(tl.dot(do.T, p, out_dtype=tl.int32).to(tl.float32), p_scale, dv_t)
                        else:
                            p_scale *= 1.0 / (65504.0 if do.dtype == tl.float16 else 448.0)
                            p_scale = tl.where(p_scale <= 2e-38, 1.0, p_scale)
                            p = tl.mul(p, tl.fdiv(1.0, p_scale)).to(do.dtype)
                            if use_fp16_accum and do.dtype == tl.float16:
                                p_scale *= fp16_scale_dv # pylint: disable=used-before-assignment,possibly-used-before-assignment
                                do_T = tl.mul(do.T.to(tl.float32), in_scale_dv).to(tl.float16) # pylint: disable=used-before-assignment,possibly-used-before-assignment
                                p = tl.mul(p.to(tl.float32), in_scale_dv).to(tl.float16) # pylint: disable=used-before-assignment,possibly-used-before-assignment
                                dv_t = tl.fma(tl.dot(do_T, p, out_dtype=tl.float16).to(tl.float32), p_scale, dv_t)
                            else:
                                dv_t = tl.fma(tl.dot(do.T, p, out_dtype=tl.float32), p_scale, dv_t)
                    else:
                        p = p.to(do.dtype)
                        dv_t = tl.dot(do.T, p, dv_t, out_dtype=tl.float32)

    if do_grad_k:
        dk_t *= sm_scale
        dk_t = dk_t.to(dk_ptr.type.element_ty)
        if use_tensor_descriptors:
            dk_desc = tl.make_tensor_descriptor(dk_ptr + offset_k * KHD, shape=[KN, KHD], strides=[KHD, 1], block_shape=[BLOCK_SIZE_N, triton.next_power_of_2(KHD)])
            dk_desc.store([start_n_block, 0], dk_t.T)
        else:
            tl.store(dk_ptr + offset_k * KHD + offs_n[None, :] * KHD + offs_khd[:, None], dk_t, mask=(offs_n[None, :] < KN) & (offs_khd[:, None] < KHD))
    if do_grad_v:
        dv_t = dv_t.to(dv_ptr.type.element_ty)
        if use_tensor_descriptors:
            dv_desc = tl.make_tensor_descriptor(dv_ptr + offset_v * VHD, shape=[VN, VHD], strides=[VHD, 1], block_shape=[BLOCK_SIZE_N, triton.next_power_of_2(VHD)])
            dv_desc.store([start_n_block, 0], dv_t.T)
        else:
            tl.store(dv_ptr + offset_v * VHD + offs_n[None, :] * VHD + offs_vhd[:, None], dv_t, mask=(offs_n[None, :] < VN) & (offs_vhd[:, None] < VHD))


@triton_op("sdnq::triton_atten_bwd_dq", mutates_args={})
@inference_context()
def sdnq_triton_atten_bwd_dq(
    grad_output: torch.Tensor,
    delta: torch.Tensor,
    lse: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output_scale: torch.Tensor | None,
    query_scale: torch.Tensor | None,
    key_scale: torch.Tensor | None,
    value_scale: torch.Tensor | None,
    attn_mask: torch.Tensor | None = None,
    sm_scale: float | None = None,
    is_causal: bool = False,
    use_fp16_accum: bool = False,
    use_tensor_descriptors: bool | None = None,
    out_dtype: torch.dtype | None = None,
    block_count: torch.Tensor | None = None,
    block_index: torch.Tensor | None = None,
    block_mask_m: int = 0,
    block_mask_n: int = 0,
) -> torch.Tensor:
    QZ, QH, QN, _ = query.shape
    _, _, KN, _ = key.shape
    _, _, VN, _ = value.shape
    (
        use_tensor_descriptors,
        use_tensor_descriptors_for_scales,
        use_tensor_descriptors_for_masks,
    ) = check_tensor_descriptors(
        query, key, value,
        grad_output=grad_output,
        delta=delta,
        lse=lse,
        grad_output_scale=grad_output_scale,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        attn_mask=attn_mask,
        use_tensor_descriptors=use_tensor_descriptors,
    )
    if block_count is not None or block_index is not None:
        check_block_lists(block_count, block_index, QZ, QH, QN, KN, block_mask_m, block_mask_n)
    else:
        block_mask_m = block_mask_n = 0
    def grid_dq(META):
        return (triton.cdiv(QN, META["BLOCK_SIZE_M"]), QH, QZ)
    dq = torch.empty_like(query, dtype=out_dtype)
    wrap_triton(sdnq_attn_bwd_dq_kernel)[grid_dq](
        query, key, value, grad_output,
        query_scale, key_scale, value_scale, grad_output_scale,
        dq, lse, delta, attn_mask, block_count, block_index, sm_scale,
        (1 if is_causal else 0),
        (1 if attn_mask is not None else 0),
        (1 if block_count is not None else 0),
        (1 if use_fp16_accum else 0),
        (1 if use_tensor_descriptors else 0),
        (1 if use_tensor_descriptors_for_scales else 0),
        (1 if use_tensor_descriptors_for_masks else 0),
        *query.shape, *key.shape, *value.shape,
        *(attn_mask.shape if attn_mask is not None else (0, 0, 0, 0)),
        block_mask_m, block_mask_n, block_count_chunk, block_index_chunk,
        *(block_count.shape[:2] if block_count is not None else (0, 0)),
        (1 if query_scale is not None else 0),
        (1 if value_scale is not None else 0),
        str(query.dtype), str(value.dtype), str(out_dtype),
        str(attn_mask.dtype if attn_mask is not None else None),
        math.ceil(QN / block_size_divisor),
        math.ceil(KN / block_size_divisor),
        math.ceil(VN / block_size_divisor),
        block_size_divisor,
    )
    return dq


@inference_context()
def sdnq_atten_bwd_dkv(
    grad_output: torch.Tensor,
    delta: torch.Tensor,
    lse: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output_scale: torch.Tensor | None,
    query_scale: torch.Tensor | None,
    key_scale: torch.Tensor | None,
    value_scale: torch.Tensor | None,
    attn_mask: torch.Tensor | None = None,
    sm_scale: float | None = None,
    is_causal: bool = False,
    use_fp16_accum: bool = False,
    use_tensor_descriptors: bool | None = None,
    out_dtype: torch.dtype | None = None,
    do_grad_k: bool = True,
    do_grad_v: bool = True,
    block_count: torch.Tensor | None = None,
    block_index: torch.Tensor | None = None,
    block_mask_m: int = 0,
    block_mask_n: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    QZ, QH, QN, _ = query.shape
    _, KH, KN, _ = key.shape
    _, _, VN, _ = value.shape
    (
        use_tensor_descriptors,
        use_tensor_descriptors_for_scales,
        use_tensor_descriptors_for_masks,
    ) = check_tensor_descriptors(
        query, key, value,
        grad_output=grad_output,
        delta=delta,
        lse=lse,
        grad_output_scale=grad_output_scale,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        attn_mask=attn_mask,
        use_tensor_descriptors=use_tensor_descriptors,
    )
    if block_count is not None or block_index is not None:
        check_block_lists(block_count, block_index, QZ, QH, KN, QN, block_mask_n, block_mask_m)
    else:
        block_mask_m = block_mask_n = 0
    def grid_dkv(META):
        return (triton.cdiv(KN, META["BLOCK_SIZE_N"]), KH, QZ)
    dk = torch.empty_like(key, dtype=out_dtype) if do_grad_k else None
    dv = torch.empty_like(value, dtype=out_dtype) if do_grad_v else None
    wrap_triton(sdnq_attn_bwd_dkv_kernel)[grid_dkv](
        query, key, value, grad_output,
        query_scale, key_scale, value_scale, grad_output_scale,
        dk, dv, lse, delta, attn_mask, block_count, block_index, sm_scale,
        (1 if do_grad_k else 0),
        (1 if do_grad_v else 0),
        (1 if is_causal else 0),
        (1 if attn_mask is not None else 0),
        (1 if block_count is not None else 0),
        (1 if use_fp16_accum else 0),
        (1 if use_tensor_descriptors else 0),
        (1 if use_tensor_descriptors_for_scales else 0),
        (1 if use_tensor_descriptors_for_masks else 0),
        *query.shape, *key.shape, *value.shape,
        *(attn_mask.shape if attn_mask is not None else (0, 0, 0, 0)),
        block_mask_m, block_mask_n, block_count_chunk, block_index_chunk,
        *(block_count.shape[:2] if block_count is not None else (0, 0)),
        (1 if query_scale is not None else 0),
        (1 if value_scale is not None else 0),
        str(query.dtype), str(value.dtype), str(out_dtype),
        str(attn_mask.dtype if attn_mask is not None else None),
        math.ceil(QN / block_size_divisor),
        math.ceil(KN / block_size_divisor),
        math.ceil(VN / block_size_divisor),
        block_size_divisor,
    )
    return dk, dv


@triton_op("sdnq::triton_atten_bwd_dkv", mutates_args={})
@inference_context()
def sdnq_triton_atten_bwd_dkv(
    grad_output: torch.Tensor,
    delta: torch.Tensor,
    lse: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output_scale: torch.Tensor | None,
    query_scale: torch.Tensor | None,
    key_scale: torch.Tensor | None,
    value_scale: torch.Tensor | None,
    attn_mask: torch.Tensor | None = None,
    sm_scale: float | None = None,
    is_causal: bool = False,
    use_fp16_accum: bool = False,
    use_tensor_descriptors: bool | None = None,
    out_dtype: torch.dtype | None = None,
    block_count: torch.Tensor | None = None,
    block_index: torch.Tensor | None = None,
    block_mask_m: int = 0,
    block_mask_n: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return sdnq_atten_bwd_dkv(
        grad_output, delta, lse,
        query, key, value,
        grad_output_scale=grad_output_scale,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        attn_mask=attn_mask,
        sm_scale=sm_scale,
        is_causal=is_causal,
        use_fp16_accum=use_fp16_accum,
        use_tensor_descriptors=use_tensor_descriptors,
        out_dtype=out_dtype,
        do_grad_k=True,
        do_grad_v=True,
        block_count=block_count,
        block_index=block_index,
        block_mask_m=block_mask_m,
        block_mask_n=block_mask_n,
    )


@triton_op("sdnq::triton_atten_bwd_dk", mutates_args={})
@inference_context()
def sdnq_triton_atten_bwd_dk(
    grad_output: torch.Tensor,
    delta: torch.Tensor,
    lse: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output_scale: torch.Tensor | None,
    query_scale: torch.Tensor | None,
    key_scale: torch.Tensor | None,
    value_scale: torch.Tensor | None,
    attn_mask: torch.Tensor | None = None,
    sm_scale: float | None = None,
    is_causal: bool = False,
    use_fp16_accum: bool = False,
    use_tensor_descriptors: bool | None = None,
    out_dtype: torch.dtype | None = None,
    block_count: torch.Tensor | None = None,
    block_index: torch.Tensor | None = None,
    block_mask_m: int = 0,
    block_mask_n: int = 0,
) -> torch.Tensor:
    return sdnq_atten_bwd_dkv(
        grad_output, delta, lse,
        query, key, value,
        grad_output_scale=grad_output_scale,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        attn_mask=attn_mask,
        sm_scale=sm_scale,
        is_causal=is_causal,
        use_fp16_accum=use_fp16_accum,
        use_tensor_descriptors=use_tensor_descriptors,
        out_dtype=out_dtype,
        do_grad_k=True,
        do_grad_v=False,
        block_count=block_count,
        block_index=block_index,
        block_mask_m=block_mask_m,
        block_mask_n=block_mask_n,
    )[0]


@triton_op("sdnq::triton_atten_bwd_dv", mutates_args={})
@inference_context()
def sdnq_triton_atten_bwd_dv(
    grad_output: torch.Tensor,
    delta: torch.Tensor,
    lse: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    grad_output_scale: torch.Tensor | None,
    query_scale: torch.Tensor | None,
    key_scale: torch.Tensor | None,
    value_scale: torch.Tensor | None,
    attn_mask: torch.Tensor | None = None,
    sm_scale: float | None = None,
    is_causal: bool = False,
    use_fp16_accum: bool = False,
    use_tensor_descriptors: bool | None = None,
    out_dtype: torch.dtype | None = None,
    block_count: torch.Tensor | None = None,
    block_index: torch.Tensor | None = None,
    block_mask_m: int = 0,
    block_mask_n: int = 0,
) -> torch.Tensor:
    return sdnq_atten_bwd_dkv(
        grad_output, delta, lse,
        query, key, value,
        grad_output_scale=grad_output_scale,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        attn_mask=attn_mask,
        sm_scale=sm_scale,
        is_causal=is_causal,
        use_fp16_accum=use_fp16_accum,
        use_tensor_descriptors=use_tensor_descriptors,
        out_dtype=out_dtype,
        do_grad_k=False,
        do_grad_v=True,
        block_count=block_count,
        block_index=block_index,
        block_mask_m=block_mask_m,
        block_mask_n=block_mask_n,
    )[1]


@inference_context()
def get_block_mask_backward_inputs(
    block_mask: torch.Tensor,
    do_grad_q: bool = True,
    do_grad_k: bool = True,
    do_grad_v: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if do_grad_q:
        block_count, block_index = get_block_mask_input(block_mask)
    else:
        block_count = block_index = None
    if do_grad_k or do_grad_v:
        block_count_t, block_index_t = get_block_mask_input(block_mask.transpose(-1, -2))
    else:
        block_count_t = block_index_t = None
    return block_count, block_index, block_count_t, block_index_t


@inference_context()
def get_attn_backward_inputs(
    grad_output: torch.Tensor,
    out: torch.Tensor,
    hadamard_pv: torch.Tensor | None,
    pv_matmul_dtype: str | None = None,
    quantize_fp32: bool = True,
    quantize_pv: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    out = out.contiguous()
    grad_output = grad_output.contiguous()
    delta = torch.sum(torch.mul(out, grad_output), dim=-1, dtype=torch.float32)
    if quantize_pv:
        if pv_matmul_dtype in {"enabled", "uint8"}:
            pv_matmul_dtype = "int8"
        if hadamard_pv is not None:
            grad_output = rotate_hadamard(grad_output, group_size=hadamard_pv.shape[-1], hadamard=hadamard_pv)
        quantize_mm_func_pv = quantize_int_mm if pv_matmul_dtype.startswith("int") else quantize_fp_mm
        if quantize_fp32:
            grad_output = grad_output.to(dtype=torch.float32)
        grad_output, grad_output_scale = quantize_mm_func_pv(grad_output, dim=-1, matmul_dtype=pv_matmul_dtype)
        grad_output_scale = grad_output_scale.squeeze(-1)
    else:
        grad_output_scale = None
    return grad_output, grad_output_scale, out, delta


@inference_context()
def sdnq_triton_atten_bwd(
    grad_output: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_scale: torch.Tensor | None,
    key_scale: torch.Tensor | None,
    value_scale: torch.Tensor | None,
    attn_mask: torch.Tensor | None = None,
    sm_scale: float | None = None,
    is_causal: bool = False,
    use_hadamard_qk: bool = False,
    hadamard_group_size_qk: int = 256,
    use_hadamard_pv: bool = False,
    hadamard_group_size_pv: int = 256,
    pv_matmul_dtype: str | None = None,
    do_quantize: bool = True,
    quantize_fp32: bool = True,
    use_fp16_accum: bool = False,
    use_tensor_descriptors: bool | None = None,
    out_dtype: torch.dtype | None = None,
    do_grad_q: bool = True,
    do_grad_k: bool = True,
    do_grad_v: bool = True,
    block_mask: torch.Tensor | None = None,
    block_mask_m: int = 0,
    block_mask_n: int = 0,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if not do_grad_q and not do_grad_k and not do_grad_v:
        return None, None, None

    if sm_scale is None:
        sm_scale = query.shape[-1] ** -0.5
    if out_dtype is None:
        out_dtype = grad_output.dtype

    if block_mask is not None:
        block_count, block_index, block_count_t, block_index_t = get_block_mask_backward_inputs(
            block_mask, do_grad_q=do_grad_q, do_grad_k=do_grad_k, do_grad_v=do_grad_v,
        )
    else:
        block_count = block_index = block_count_t = block_index_t = None

    quantize_pv = bool(do_quantize and pv_matmul_dtype not in {None, "auto", "none", "no", "disabled"})

    if use_hadamard_qk and (do_grad_q or do_grad_k):
        hadamard_qk = get_hadamard(hadamard_group_size_qk, dtype=grad_output.dtype, device=grad_output.device)
    else:
        hadamard_qk = None
    if use_hadamard_pv:
        hadamard_pv = get_hadamard(hadamard_group_size_pv, dtype=grad_output.dtype, device=grad_output.device)
    else:
        hadamard_pv = None

    grad_output, grad_output_scale, out, delta = get_attn_backward_inputs(
        grad_output, out,
        hadamard_pv=hadamard_pv, pv_matmul_dtype=pv_matmul_dtype,
        quantize_fp32=quantize_fp32, quantize_pv=quantize_pv,
    )

    if do_grad_q:
        dq = sdnq_triton_atten_bwd_dq(
            grad_output, delta, lse,
            query, key, value,
            grad_output_scale=grad_output_scale,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            attn_mask=attn_mask,
            sm_scale=sm_scale,
            is_causal=is_causal,
            use_fp16_accum=use_fp16_accum,
            use_tensor_descriptors=use_tensor_descriptors,
            out_dtype=out_dtype,
            block_count=block_count,
            block_index=block_index,
            block_mask_m=block_mask_m,
            block_mask_n=block_mask_n,
        )
        if use_hadamard_qk:
            dq = rotate_hadamard_compiled(dq, group_size=hadamard_group_size_qk, hadamard=hadamard_qk)
    else:
        dq = None

    if do_grad_k and do_grad_v:
        dk, dv = sdnq_triton_atten_bwd_dkv(
            grad_output, delta, lse,
            query, key, value,
            grad_output_scale=grad_output_scale,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            attn_mask=attn_mask,
            sm_scale=sm_scale,
            is_causal=is_causal,
            use_fp16_accum=use_fp16_accum,
            use_tensor_descriptors=use_tensor_descriptors,
            out_dtype=out_dtype,
            block_count=block_count_t,
            block_index=block_index_t,
            block_mask_m=block_mask_m,
            block_mask_n=block_mask_n,
        )
    elif do_grad_k:
        dv = None
        dk = sdnq_triton_atten_bwd_dk(
            grad_output, delta, lse,
            query, key, value,
            grad_output_scale=grad_output_scale,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            attn_mask=attn_mask,
            sm_scale=sm_scale,
            is_causal=is_causal,
            use_fp16_accum=use_fp16_accum,
            use_tensor_descriptors=use_tensor_descriptors,
            out_dtype=out_dtype,
            block_count=block_count_t,
            block_index=block_index_t,
            block_mask_m=block_mask_m,
            block_mask_n=block_mask_n,
        )
    elif do_grad_v:
        dk = None
        dv = sdnq_triton_atten_bwd_dv(
            grad_output, delta, lse,
            query, key, value,
            grad_output_scale=grad_output_scale,
            query_scale=query_scale,
            key_scale=key_scale,
            value_scale=value_scale,
            attn_mask=attn_mask,
            sm_scale=sm_scale,
            is_causal=is_causal,
            use_fp16_accum=use_fp16_accum,
            use_tensor_descriptors=use_tensor_descriptors,
            out_dtype=out_dtype,
            block_count=block_count_t,
            block_index=block_index_t,
            block_mask_m=block_mask_m,
            block_mask_n=block_mask_n,
        )
    else:
        dk = dv = None

    if do_grad_k and use_hadamard_qk:
        dk = rotate_hadamard_compiled(dk, group_size=hadamard_group_size_qk, hadamard=hadamard_qk)
    if do_grad_v and use_hadamard_pv:
        dv = rotate_hadamard_compiled(dv, group_size=hadamard_group_size_pv, hadamard=hadamard_pv)

    return dq, dk, dv


class SDNQAttenBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        scale: float | None,
        smooth_k: bool,
        use_hadamard: bool,
        hadamard_group_size: int,
        matmul_dtype: str,
        pv_matmul_dtype: str | None,
        do_quantize: bool,
        quantize_fp32: bool,
        use_fp16_accum: bool,
        use_tensor_descriptors: bool,
        out_dtype: torch.dtype | None,
        block_mask: torch.Tensor | None = None,
        block_mask_m: int = 0,
        block_mask_n: int = 0,
    ) -> torch.Tensor:
        ctx.is_causal = is_causal
        ctx.do_quantize = do_quantize
        ctx.quantize_fp32 = quantize_fp32
        ctx.use_fp16_accum = use_fp16_accum
        ctx.use_tensor_descriptors = use_tensor_descriptors
        ctx.pv_matmul_dtype = pv_matmul_dtype
        ctx.block_mask_m = block_mask_m
        ctx.block_mask_n = block_mask_n

        (
            out, lse, query, key, value,
            query_scale, key_scale, value_scale,
            attn_mask, block_mask, sm_scale,
            use_hadamard_qk, hadamard_group_size_qk,
            use_hadamard_pv, hadamard_group_size_pv,
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
            quantize_fp32=quantize_fp32,
            use_fp16_accum=use_fp16_accum,
            use_tensor_descriptors=use_tensor_descriptors,
            out_dtype=out_dtype,
            return_backward=True,
            block_mask=block_mask,
            block_mask_m=block_mask_m,
            block_mask_n=block_mask_n,
        )

        ctx.sm_scale = sm_scale
        ctx.use_hadamard_qk = use_hadamard_qk
        ctx.hadamard_group_size_qk = hadamard_group_size_qk
        ctx.use_hadamard_pv = use_hadamard_pv
        ctx.hadamard_group_size_pv = hadamard_group_size_pv
        ctx.save_for_backward(out, lse, query, key, value, query_scale, key_scale, value_scale, attn_mask, block_mask)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        out, lse, query, key, value, query_scale, key_scale, value_scale, attn_mask, block_mask = ctx.saved_tensors
        dq, dk, dv = sdnq_triton_atten_bwd(
            grad_output, out, lse, query, key, value,
            query_scale, key_scale, value_scale,
            attn_mask=attn_mask,
            sm_scale=ctx.sm_scale,
            pv_matmul_dtype=ctx.pv_matmul_dtype,
            is_causal=ctx.is_causal,
            use_hadamard_qk=ctx.use_hadamard_qk,
            hadamard_group_size_qk=ctx.hadamard_group_size_qk,
            use_hadamard_pv=ctx.use_hadamard_pv,
            hadamard_group_size_pv=ctx.hadamard_group_size_pv,
            do_quantize=ctx.do_quantize,
            quantize_fp32=ctx.do_quantize,
            use_fp16_accum=ctx.use_fp16_accum,
            use_tensor_descriptors=ctx.use_tensor_descriptors,
            do_grad_q=ctx.needs_input_grad[0],
            do_grad_k=ctx.needs_input_grad[1],
            do_grad_v=ctx.needs_input_grad[2],
            block_mask=block_mask,
            block_mask_m=ctx.block_mask_m,
            block_mask_n=ctx.block_mask_n,
        )
        return dq, dk, dv, *(None,)*16


def sdnq_triton_atten_with_backward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
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
        quantize_fp32: bool = True,
        use_fp16_accum: bool = False,
        use_tensor_descriptors: bool | None = None,
        out_dtype: torch.dtype | None = None,
        block_mask: torch.Tensor | None = None,
        block_mask_m: int = 0,
        block_mask_n: int = 0,
    ) -> torch.Tensor:
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
        quantize_fp32,
        use_fp16_accum,
        use_tensor_descriptors,
        out_dtype,
        block_mask,
        block_mask_m,
        block_mask_n,
    )


get_attn_backward_inputs = compile_func(get_attn_backward_inputs)
get_block_mask_backward_inputs = compile_func(get_block_mask_backward_inputs)
