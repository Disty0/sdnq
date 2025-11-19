from typing import Optional, Union

import time
import torch
from tqdm import tqdm

import sdnq.common
import sdnq.quantizer
from sdnq import sdnq_quantize_layer


def get_tflops(it_s: float, m: int, n: int, k: int) -> float:
    return round(it_s * ((2*m*k*n) + (n * m)) / (10**12), 2)


def benchmark_linear(name: str, linear: torch.nn.Linear, x: torch.Tensor, steps: int):
    assert x.ndim == 2
    try:
        print(name)
        sync_func = getattr(torch, x.device.type).synchronize
        z = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            z = linear(x)
            sync_func()
        t1 = time.time()
        return get_tflops(steps/(t1 - t0), x.shape[0],z.shape[1],x.shape[1])
    except Exception:
        print(f"{name} test failed")
        return 0


@torch.no_grad()
def main(
    steps: int = 50,
    mnk: int = 8192,
    dtype: Optional[Union[torch.dtype, str]] = None,
    device: Optional[str] = None,
    m: Optional[int] = None,
    n: Optional[int] = None,
    k: Optional[int] = None,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else None
        if device is None:
            raise RuntimeError("A GPU is required to run SDNQ Benchmark")

    if dtype is None:
        dtype = torch.bfloat16 if not sdnq.common.is_rdna2 else torch.float16
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if m is None:
        m = 2*mnk
    if n is None:
        n = mnk
    if k is None:
        k = mnk//2

    x = torch.randn(m,k, device=device, dtype=dtype)
    x.requires_grad_(False)


    pytorch_float_tflops = benchmark_linear("PyTorch Float", torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), x, steps)

    if sdnq.common.use_torch_compile:
        sdnq_int8_tflops = benchmark_linear("SDNQ INT8", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int8", torch_dtype=dtype, use_quantized_matmul=True), x, steps)
        sdnq_int8_svd_tflops = benchmark_linear("SDNQ INT8 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int8", torch_dtype=dtype, use_quantized_matmul=True, use_svd=True), x, steps)

        backup_tw_fp8 = sdnq.common.use_tensorwise_fp8_matmul

        sdnq.common.use_tensorwise_fp8_matmul = False
        sdnq.quantizer.use_tensorwise_fp8_matmul = False
        sdnq.forward.use_tensorwise_fp8_matmul = False
        sdnq_fp8_tflops = benchmark_linear("SDNQ FP8", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e4m3fn", torch_dtype=dtype, use_quantized_matmul=True), x, steps)
        sdnq_fp8_svd_tflops = benchmark_linear("SDNQ FP8 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e4m3fn", torch_dtype=dtype, use_quantized_matmul=True, use_svd=True), x, steps)

        sdnq.common.use_tensorwise_fp8_matmul = True
        sdnq.quantizer.use_tensorwise_fp8_matmul = True
        sdnq.forward.use_tensorwise_fp8_matmul = True
        sdnq_fp8_tw_tflops = benchmark_linear("SDNQ FP8 TW", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e4m3fn", torch_dtype=dtype, use_quantized_matmul=True), x, steps)
        sdnq_fp8_tw_svd_tflops = benchmark_linear("SDNQ FP8 TW SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e4m3fn", torch_dtype=dtype, use_quantized_matmul=True, use_svd=True), x, steps)

        sdnq.common.use_tensorwise_fp8_matmul = backup_tw_fp8
        sdnq.quantizer.use_tensorwise_fp8_matmul = backup_tw_fp8
        sdnq.forward.use_tensorwise_fp8_matmul = backup_tw_fp8

        sdnq_int8_int7_tflops = benchmark_linear("SDNQ INT8 INT7", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int7", torch_dtype=dtype, use_quantized_matmul=True), x, steps)
        sdnq_int8_int5_tflops = benchmark_linear("SDNQ INT8 INT6", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int6", torch_dtype=dtype, use_quantized_matmul=True), x, steps)
        sdnq_int8_int6_tflops = benchmark_linear("SDNQ INT8 INT5", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int5", torch_dtype=dtype, use_quantized_matmul=True), x, steps)
        sdnq_int8_uint4_tflops = benchmark_linear("SDNQ INT8 UINT4", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint4", torch_dtype=dtype, use_quantized_matmul=True), x, steps)
        sdnq_int8_uint3_tflops = benchmark_linear("SDNQ INT8 UINT3", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint3", torch_dtype=dtype, use_quantized_matmul=True), x, steps)
        sdnq_int8_uint2_tflops = benchmark_linear("SDNQ INT8 UINT2", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint2", torch_dtype=dtype, use_quantized_matmul=True), x, steps)
        sdnq_int8_uint1_tflops = benchmark_linear("SDNQ INT8 UINT1", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint1", torch_dtype=dtype, use_quantized_matmul=True), x, steps)

        sdnq_int8_int7_svd_tflops = benchmark_linear("SDNQ INT8 INT7 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int7", torch_dtype=dtype, use_quantized_matmul=True, use_svd=True), x, steps)
        sdnq_int8_int5_svd_tflops = benchmark_linear("SDNQ INT8 INT6 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int6", torch_dtype=dtype, use_quantized_matmul=True, use_svd=True), x, steps)
        sdnq_int8_int6_svd_tflops = benchmark_linear("SDNQ INT8 INT5 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int5", torch_dtype=dtype, use_quantized_matmul=True, use_svd=True), x, steps)
        sdnq_int8_uint4_svd_tflops = benchmark_linear("SDNQ INT8 UINT4 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint4", torch_dtype=dtype, use_quantized_matmul=True, use_svd=True), x, steps)
        sdnq_int8_uint3_svd_tflops = benchmark_linear("SDNQ INT8 UINT3 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint3", torch_dtype=dtype, use_quantized_matmul=True, use_svd=True), x, steps)
        sdnq_int8_uint2_svd_tflops = benchmark_linear("SDNQ INT8 UINT2 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint2", torch_dtype=dtype, use_quantized_matmul=True, use_svd=True), x, steps)
        sdnq_int8_uint1_svd_tflops = benchmark_linear("SDNQ INT8 UINT1 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint1", torch_dtype=dtype, use_quantized_matmul=True, use_svd=True), x, steps)
    else:
        print("Torch Compile is disabled, skipping quantized matmul tests.")

    sdnq_float_int16_tflops = benchmark_linear("SDNQ Float INT16", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int16", torch_dtype=dtype, use_quantized_matmul=False), x, steps)
    sdnq_float_int8_tflops = benchmark_linear("SDNQ Float INT8", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int8", torch_dtype=dtype, use_quantized_matmul=False), x, steps)
    sdnq_float_int7_tflops = benchmark_linear("SDNQ Float INT7", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int7", torch_dtype=dtype, use_quantized_matmul=False), x, steps)
    sdnq_float_int6_tflops = benchmark_linear("SDNQ Float INT6", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int6", torch_dtype=dtype, use_quantized_matmul=False), x, steps)
    sdnq_float_int5_tflops = benchmark_linear("SDNQ Float INT5", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int5", torch_dtype=dtype, use_quantized_matmul=False), x, steps)
    sdnq_float_uint4_tflops = benchmark_linear("SDNQ Float UINT4", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint4", torch_dtype=dtype, use_quantized_matmul=False), x, steps)
    sdnq_float_uint3_tflops = benchmark_linear("SDNQ Float UINT3", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint3", torch_dtype=dtype, use_quantized_matmul=False), x, steps)
    sdnq_float_uint2_tflops = benchmark_linear("SDNQ Float UINT2", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint2", torch_dtype=dtype, use_quantized_matmul=False), x, steps)
    sdnq_float_uint1_tflops = benchmark_linear("SDNQ Float UINT1", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint1", torch_dtype=dtype, use_quantized_matmul=False), x, steps)

    sdnq_float_fp16_tflops = benchmark_linear("SDNQ Float FP16", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float16", torch_dtype=dtype, use_quantized_matmul=False), x, steps)
    sdnq_float_fp8_e4_tflops = benchmark_linear("SDNQ Float FP8 E4", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e4m3fn", torch_dtype=dtype, use_quantized_matmul=False), x, steps)
    sdnq_float_fp8_e5_tflops = benchmark_linear("SDNQ Float FP8 E5", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e5m2", torch_dtype=dtype, use_quantized_matmul=False), x, steps)
    sdnq_float_fp8_e4fnuz_tflops = benchmark_linear("SDNQ Float FP8 E4 FNUZ", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e4m3fnuz", torch_dtype=dtype, use_quantized_matmul=False), x, steps)
    sdnq_float_fp8_e5fnuz_tflops = benchmark_linear("SDNQ Float FP8 E5 FNUZ", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e5m2fnuz", torch_dtype=dtype, use_quantized_matmul=False), x, steps)

    sdnq_float_int16_svd_tflops = benchmark_linear("SDNQ Float INT16 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int16", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)
    sdnq_float_int8_svd_tflops = benchmark_linear("SDNQ Float INT8 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int8", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)
    sdnq_float_int7_svd_tflops = benchmark_linear("SDNQ Float INT7 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int7", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)
    sdnq_float_int6_svd_tflops = benchmark_linear("SDNQ Float INT6 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int6", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)
    sdnq_float_int5_svd_tflops = benchmark_linear("SDNQ Float INT5 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int5", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)
    sdnq_float_uint4_svd_tflops = benchmark_linear("SDNQ Float UINT4 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint4", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)
    sdnq_float_uint3_svd_tflops = benchmark_linear("SDNQ Float UINT3 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint3", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)
    sdnq_float_uint2_svd_tflops = benchmark_linear("SDNQ Float UINT2 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint2", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)
    sdnq_float_uint1_svd_tflops = benchmark_linear("SDNQ Float UINT1 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint1", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)

    sdnq_float_fp16_svd_tflops = benchmark_linear("SDNQ Float FP16 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float16", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)
    sdnq_float_fp8_e4_svd_tflops = benchmark_linear("SDNQ Float FP8 E4 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e4m3fn", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)
    sdnq_float_fp8_e5_svd_tflops = benchmark_linear("SDNQ Float FP8 E5 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e5m2", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)
    sdnq_float_fp8_e4fnuz_svd_tflops = benchmark_linear("SDNQ Float FP8 E4 FNUZ SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e4m3fnuz", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)
    sdnq_float_fp8_e5fnuz_svd_tflops = benchmark_linear("SDNQ Float FP8 E5 FNUZ SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e5m2fnuz", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True), x, steps)

    print("")
    print("==================================================")
    print("GPU:", getattr(torch, torch.device(device).type).get_device_name(device))
    print("Steps:", steps, "| MNK:", round((m*n*k)**(1/3)), "| Float:", dtype)
    print("M:", m, "| N:", n, "| K:", k)
    print("Torch Compile:", sdnq.common.use_torch_compile)
    print("Contiguous MM:", sdnq.common.use_contiguous_mm)
    print("Triton MM:", sdnq.common.use_triton_mm)
    print("==================================================")
    print("PyTorch Float TFLOPS:", pytorch_float_tflops)
    if sdnq.common.use_torch_compile:
        print("==================================================")
        print("SDNQ INT8 TFLOPS:", sdnq_int8_tflops)
        print("SDNQ FP8 TFLOPS:", sdnq_fp8_tflops)
        print("SDNQ FP8 TW TFLOPS:", sdnq_fp8_tw_tflops)
        print("==================================================")
        print("SDNQ INT8 SVD TFLOPS:", sdnq_int8_svd_tflops)
        print("SDNQ FP8 SVD TFLOPS:", sdnq_fp8_svd_tflops)
        print("SDNQ FP8 TW SVD TFLOPS:", sdnq_fp8_tw_svd_tflops)
        print("==================================================")
        print("SDNQ INT8 INT7 TFLOPS:", sdnq_int8_int7_tflops)
        print("SDNQ INT8 INT6 TFLOPS:", sdnq_int8_int6_tflops)
        print("SDNQ INT8 INT5 TFLOPS:", sdnq_int8_int5_tflops)
        print("SDNQ INT8 UINT4 TFLOPS:", sdnq_int8_uint4_tflops)
        print("SDNQ INT8 UINT3 TFLOPS:", sdnq_int8_uint3_tflops)
        print("SDNQ INT8 UINT2 TFLOPS:", sdnq_int8_uint2_tflops)
        print("SDNQ INT8 UINT1 TFLOPS:", sdnq_int8_uint1_tflops)
        print("==================================================")
        print("SDNQ INT8 INT7 SVD TFLOPS:", sdnq_int8_int7_svd_tflops)
        print("SDNQ INT8 INT6 SVD TFLOPS:", sdnq_int8_int6_svd_tflops)
        print("SDNQ INT8 INT5 SVD TFLOPS:", sdnq_int8_int5_svd_tflops)
        print("SDNQ INT8 UINT4 SVD TFLOPS:", sdnq_int8_uint4_svd_tflops)
        print("SDNQ INT8 UINT3 SVD TFLOPS:", sdnq_int8_uint3_svd_tflops)
        print("SDNQ INT8 UINT2 SVD TFLOPS:", sdnq_int8_uint2_svd_tflops)
        print("SDNQ INT8 UINT1 SVD TFLOPS:", sdnq_int8_uint1_svd_tflops)
    print("==================================================")
    print("SDNQ Float INT16 TFLOPS:", sdnq_float_int16_tflops)
    print("SDNQ Float INT8 TFLOPS:", sdnq_float_int8_tflops)
    print("SDNQ Float INT7 TFLOPS:", sdnq_float_int7_tflops)
    print("SDNQ Float INT6 TFLOPS:", sdnq_float_int6_tflops)
    print("SDNQ Float INT5 TFLOPS:", sdnq_float_int5_tflops)
    print("SDNQ Float UINT4 TFLOPS:", sdnq_float_uint4_tflops)
    print("SDNQ Float UINT3 TFLOPS:", sdnq_float_uint3_tflops)
    print("SDNQ Float UINT2 TFLOPS:", sdnq_float_uint2_tflops)
    print("SDNQ Float UINT1 TFLOPS:", sdnq_float_uint1_tflops)
    print("==================================================")
    print("SDNQ Float INT16 SVD TFLOPS:", sdnq_float_int16_svd_tflops)
    print("SDNQ Float INT8 SVD TFLOPS:", sdnq_float_int8_svd_tflops)
    print("SDNQ Float INT7 SVD TFLOPS:", sdnq_float_int7_svd_tflops)
    print("SDNQ Float INT6 SVD TFLOPS:", sdnq_float_int6_svd_tflops)
    print("SDNQ Float INT5 SVD TFLOPS:", sdnq_float_int5_svd_tflops)
    print("SDNQ Float UINT4 SVD TFLOPS:", sdnq_float_uint4_svd_tflops)
    print("SDNQ Float UINT3 SVD TFLOPS:", sdnq_float_uint3_svd_tflops)
    print("SDNQ Float UINT2 SVD TFLOPS:", sdnq_float_uint2_svd_tflops)
    print("SDNQ Float UINT1 SVD TFLOPS:", sdnq_float_uint1_svd_tflops)
    print("==================================================")
    print("SDNQ Float FP16 TFLOPS:", sdnq_float_fp16_tflops)
    print("SDNQ Float FP8 E4 TFLOPS:", sdnq_float_fp8_e4_tflops)
    print("SDNQ Float FP8 E5 TFLOPS:", sdnq_float_fp8_e5_tflops)
    print("SDNQ Float FP8 E4 FNUZ TFLOPS:", sdnq_float_fp8_e4fnuz_tflops)
    print("SDNQ Float FP8 E5 FNUZ TFLOPS:", sdnq_float_fp8_e5fnuz_tflops)
    print("==================================================")
    print("SDNQ Float FP16 SVD TFLOPS:", sdnq_float_fp16_svd_tflops)
    print("SDNQ Float FP8 E4 SVD TFLOPS:", sdnq_float_fp8_e4_svd_tflops)
    print("SDNQ Float FP8 E5 SVD TFLOPS:", sdnq_float_fp8_e5_svd_tflops)
    print("SDNQ Float FP8 E4 FNUZ SVD TFLOPS:", sdnq_float_fp8_e4fnuz_svd_tflops)
    print("SDNQ Float FP8 E5 FNUZ SVD TFLOPS:", sdnq_float_fp8_e5fnuz_svd_tflops)
    print("==================================================")
    print("")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Create a bucket list with a given dataset path')
    parser.add_argument('--steps', default=100, type=int)
    parser.add_argument('--mnk', default=8192, type=int)
    parser.add_argument('--dtype', default=None, type=str)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('-m', default=None, type=int)
    parser.add_argument('-n', default=None, type=int)
    parser.add_argument('-k', default=None, type=int)

    args = parser.parse_args()
    main(steps=args.steps, mnk=args.mnk, dtype=args.dtype, device=args.device, m=args.m, n=args.n, k=args.k)
