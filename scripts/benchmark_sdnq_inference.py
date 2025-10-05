from typing import Optional, Union

import time
import torch
from tqdm import tqdm

import sdnq.common
from sdnq import sdnq_quantize_layer


def get_tflops(it_s: float, m: int, n: int, k: int) -> float:
    return round(it_s * ((2*m*k*n) + (n * m)) / (10**12), 2)


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
    sync_func = getattr(torch, torch.device(device).type).synchronize

    if dtype is None:
        dtype = torch.bfloat16
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if m is None:
        m = 2*mnk
    if n is None:
        n = mnk
    if k is None:
        k = mnk//2

    x = torch.randn(m,k, device=device, dtype=dtype)

    try:
        print("PyTorch Float:")
        linear = torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        pytorch_float_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("PyTorch Float test failed")
        pytorch_float_tflops = 0


    if sdnq.common.use_torch_compile:
        try:
            print("SDNQ INT8:")
            linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int8", torch_dtype=dtype, use_quantized_matmul=True)
            _ = linear(x)
            sync_func()
            t0 = time.time()
            for i in tqdm(range(steps)):
                _ = linear(x)
                sync_func()
            t1 = time.time()
            sdnq_int8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
        except Exception:
            print("SDNQ INT8 test failed")
            sdnq_int8_tflops = 0


        backup_tw_fp8 = sdnq.common.use_tensorwise_fp8_matmul
        try:
            print("SDNQ FP8:")
            sdnq.common.use_tensorwise_fp8_matmul = False
            linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp8", torch_dtype=dtype, use_quantized_matmul=True)
            _ = linear(x)
            sync_func()
            t0 = time.time()
            for i in tqdm(range(steps)):
                _ = linear(x)
                sync_func()
            t1 = time.time()
            sdnq_fp8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
        except Exception:
            print("SDNQ FP8 test failed")
            sdnq_fp8_tflops = 0
        sdnq.common.use_tensorwise_fp8_matmul = backup_tw_fp8


        backup_tw_fp8 = sdnq.common.use_tensorwise_fp8_matmul
        try:
            print("SDNQ FP8 TW:")
            sdnq.common.use_tensorwise_fp8_matmul = True
            linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp8", torch_dtype=dtype, use_quantized_matmul=True)
            _ = linear(x)
            sync_func()
            t0 = time.time()
            for i in tqdm(range(steps)):
                _ = linear(x)
                sync_func()
            t1 = time.time()
            sdnq_fp8_tw_tflops = get_tflops(steps/(t1 - t0), m,n,k)
        except Exception:
            print("SDNQ FP8 TW test failed")
            sdnq_fp8_tw_tflops = 0
        sdnq.common.use_tensorwise_fp8_matmul = backup_tw_fp8


        try:
            print("SDNQ INT8 INT7:")
            linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int7", torch_dtype=dtype, use_quantized_matmul=True)
            _ = linear(x)
            sync_func()
            t0 = time.time()
            for i in tqdm(range(steps)):
                _ = linear(x)
                sync_func()
            t1 = time.time()
            sdnq_int8_int7_tflops = get_tflops(steps/(t1 - t0), m,n,k)
        except Exception:
            print("SDNQ INT8 INT7 test failed")
            sdnq_int8_int7_tflops = 0


        try:
            print("SDNQ INT8 INT6:")
            linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int6", torch_dtype=dtype, use_quantized_matmul=True)
            _ = linear(x)
            sync_func()
            t0 = time.time()
            for i in tqdm(range(steps)):
                _ = linear(x)
                sync_func()
            t1 = time.time()
            sdnq_int8_int6_tflops = get_tflops(steps/(t1 - t0), m,n,k)
        except Exception:
            print("SDNQ INT8 INT6 test failed")
            sdnq_int8_int6_tflops = 0


        try:
            print("SDNQ INT8 INT5:")
            linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int5", torch_dtype=dtype, use_quantized_matmul=True)
            _ = linear(x)
            sync_func()
            t0 = time.time()
            for i in tqdm(range(steps)):
                _ = linear(x)
                sync_func()
            t1 = time.time()
            sdnq_int8_int5_tflops = get_tflops(steps/(t1 - t0), m,n,k)
        except Exception:
            print("SDNQ INT8 INT5 test failed")
            sdnq_int8_int5_tflops = 0


        try:
            print("SDNQ INT8 UINT4:")
            linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint4", torch_dtype=dtype, use_quantized_matmul=True)
            _ = linear(x)
            sync_func()
            t0 = time.time()
            for i in tqdm(range(steps)):
                _ = linear(x)
                sync_func()
            t1 = time.time()
            sdnq_int8_uint4_tflops = get_tflops(steps/(t1 - t0), m,n,k)
        except Exception:
            print("SDNQ INT8 UINT4 test failed")
            sdnq_int8_uint4_tflops = 0


        try:
            print("SDNQ INT8 UINT3:")
            linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint3", torch_dtype=dtype, use_quantized_matmul=True)
            _ = linear(x)
            sync_func()
            t0 = time.time()
            for i in tqdm(range(steps)):
                _ = linear(x)
                sync_func()
            t1 = time.time()
            sdnq_int8_uint3_tflops = get_tflops(steps/(t1 - t0), m,n,k)
        except Exception:
            print("SDNQ INT8 UINT3 test failed")
            sdnq_int8_uint3_tflops = 0


        try:
            print("SDNQ INT8 UINT2:")
            linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint2", torch_dtype=dtype, use_quantized_matmul=True)
            _ = linear(x)
            sync_func()
            t0 = time.time()
            for i in tqdm(range(steps)):
                _ = linear(x)
                sync_func()
            t1 = time.time()
            sdnq_int8_uint2_tflops = get_tflops(steps/(t1 - t0), m,n,k)
        except Exception:
            print("SDNQ INT8 UINT2 test failed")
            sdnq_int8_uint2_tflops = 0


        try:
            print("SDNQ INT8 UINT1:")
            linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint1", torch_dtype=dtype, use_quantized_matmul=True)
            _ = linear(x)
            sync_func()
            t0 = time.time()
            for i in tqdm(range(steps)):
                _ = linear(x)
                sync_func()
            t1 = time.time()
            sdnq_int8_uint1_tflops = get_tflops(steps/(t1 - t0), m,n,k)
        except Exception:
            print("SDNQ INT8 UINT1 test failed")
            sdnq_int8_uint1_tflops = 0
    else:
        print("Torch Compile is disabled, skipping quantized matmul tests.")


    try:
        print("SDNQ Float INT8:")
        linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int8", torch_dtype=dtype, use_quantized_matmul=False)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        sdnq_float_int8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("SDNQ Float INT8 test failed")
        sdnq_float_int8_tflops = 0


    try:
        print("SDNQ Float INT7:")
        linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int7", torch_dtype=dtype, use_quantized_matmul=False)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        sdnq_float_int7_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("SDNQ Float INT7 test failed")
        sdnq_float_int7_tflops = 0


    try:
        print("SDNQ Float INT6:")
        linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int6", torch_dtype=dtype, use_quantized_matmul=False)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        sdnq_float_int6_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("SDNQ Float INT6 test failed")
        sdnq_float_int6_tflops = 0


    try:
        print("SDNQ Float INT5:")
        linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int5", torch_dtype=dtype, use_quantized_matmul=False)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        sdnq_float_int5_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("SDNQ Float INT5 test failed")
        sdnq_float_int5_tflops = 0


    try:
        print("SDNQ Float UINT4:")
        linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint4", torch_dtype=dtype, use_quantized_matmul=False)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        sdnq_float_uint4_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("SDNQ Float UINT4 test failed")
        sdnq_float_uint4_tflops = 0


    try:
        print("SDNQ Float UINT3:")
        linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint3", torch_dtype=dtype, use_quantized_matmul=False)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        sdnq_float_uint3_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("SDNQ Float UINT3 test failed")
        sdnq_float_uint3_tflops = 0


    try:
        print("SDNQ Float UINT2:")
        linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint2", torch_dtype=dtype, use_quantized_matmul=False)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        sdnq_float_uint2_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("SDNQ Float UINT2 test failed")
        sdnq_float_uint2_tflops = 0


    try:
        print("SDNQ Float UINT1:")
        linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint1", torch_dtype=dtype, use_quantized_matmul=False)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        sdnq_float_uint1_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("SDNQ Float UINT1 test failed")
        sdnq_float_uint1_tflops = 0


    try:
        print("SDNQ Float FP8 E4:")
        linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e4m3fn", torch_dtype=dtype, use_quantized_matmul=False)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        sdnq_float_fp8_e4_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("SDNQ Float FP8 E4 test failed")
        sdnq_float_fp8_e4_tflops = 0


    try:
        print("SDNQ Float FP8 E5:")
        linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e5m2", torch_dtype=dtype, use_quantized_matmul=False)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        sdnq_float_fp8_e5_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("SDNQ Float FP8 E5 test failed")
        sdnq_float_fp8_e5_tflops = 0


    try:
        print("SDNQ Float FP8 E4 FNUZ:")
        linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e4m3fnuz", torch_dtype=dtype, use_quantized_matmul=False)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        sdnq_float_fp8_e4fnuz_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("SDNQ Float FP8 E4 FNUZ test failed")
        sdnq_float_fp8_e4fnuz_tflops = 0


    try:
        print("SDNQ Float FP8 E5 FNUZ:")
        linear = sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="float8_e5m2fnuz", torch_dtype=dtype, use_quantized_matmul=False)
        _ = linear(x)
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            _ = linear(x)
            sync_func()
        t1 = time.time()
        sdnq_float_fp8_e5fnuz_tflops = get_tflops(steps/(t1 - t0), m,n,k)
    except Exception:
        print("SDNQ Float FP8 E5 FNUZ test failed")
        sdnq_float_fp8_e5fnuz_tflops = 0


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
        print("SDNQ INT8 INT7 TFLOPS:", sdnq_int8_int7_tflops)
        print("SDNQ INT8 INT6 TFLOPS:", sdnq_int8_int6_tflops)
        print("SDNQ INT8 INT5 TFLOPS:", sdnq_int8_int5_tflops)
        print("SDNQ INT8 UINT4 TFLOPS:", sdnq_int8_uint4_tflops)
        print("SDNQ INT8 UINT3 TFLOPS:", sdnq_int8_uint3_tflops)
        print("SDNQ INT8 UINT2 TFLOPS:", sdnq_int8_uint2_tflops)
        print("SDNQ INT8 UINT1 TFLOPS:", sdnq_int8_uint1_tflops)
    print("==================================================")
    print("SDNQ Float INT8 TFLOPS:", sdnq_float_int8_tflops)
    print("SDNQ Float INT7 TFLOPS:", sdnq_float_int7_tflops)
    print("SDNQ Float INT6 TFLOPS:", sdnq_float_int6_tflops)
    print("SDNQ Float INT5 TFLOPS:", sdnq_float_int5_tflops)
    print("SDNQ Float UINT4 TFLOPS:", sdnq_float_uint4_tflops)
    print("SDNQ Float UINT3 TFLOPS:", sdnq_float_uint3_tflops)
    print("SDNQ Float UINT2 TFLOPS:", sdnq_float_uint2_tflops)
    print("SDNQ Float UINT1 TFLOPS:", sdnq_float_uint1_tflops)
    print("==================================================")
    print("SDNQ Float FP8 E4 TFLOPS:", sdnq_float_fp8_e4_tflops)
    print("SDNQ Float FP8 E5 TFLOPS:", sdnq_float_fp8_e5_tflops)
    print("SDNQ Float FP8 E4 FNUZ TFLOPS:", sdnq_float_fp8_e4fnuz_tflops)
    print("SDNQ Float FP8 E5 FNUZ TFLOPS:", sdnq_float_fp8_e5fnuz_tflops)
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
