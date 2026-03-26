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
    dtype: torch.dtype | str = None,
    device: str = None,
    m: int = None,
    n: int = None,
    k: int = None,
) -> None:
    if device is None:
        from sdnq.sdnext import devices
        device = devices.device

    if dtype is None:
        from sdnq.sdnext import devices
        dtype = devices.dtype
    elif isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    if m is None:
        m = 2*mnk
    if n is None:
        n = mnk
    if k is None:
        k = mnk//2

    x = torch.randn(m,k, device=device, dtype=dtype).requires_grad_(False)

    pytorch_float_tflops = benchmark_linear("PyTorch Float", torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), x, steps)

    if sdnq.common.use_torch_compile:
        sdnq_int8_tflops = benchmark_linear("SDNQ INT8", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1)[0], x, steps)
        sdnq_int8_svd_tflops = benchmark_linear("SDNQ INT8 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1, use_svd=True)[0], x, steps)

        sdnq_fp16_tflops = benchmark_linear("SDNQ FP16", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp16", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1)[0], x, steps)
        sdnq_fp16_svd_tflops = benchmark_linear("SDNQ FP16 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp16", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1, use_svd=True)[0], x, steps)

        backup_tw_fp8 = sdnq.common.use_tensorwise_fp8_matmul

        sdnq.common.use_tensorwise_fp8_matmul = False
        sdnq.quantizer.use_tensorwise_fp8_matmul = False
        sdnq.forward.use_tensorwise_fp8_matmul = False
        sdnq_fp8_tflops = benchmark_linear("SDNQ FP8", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1)[0], x, steps)
        sdnq_fp8_svd_tflops = benchmark_linear("SDNQ FP8 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1, use_svd=True)[0], x, steps)

        sdnq.common.use_tensorwise_fp8_matmul = True
        sdnq.quantizer.use_tensorwise_fp8_matmul = True
        sdnq.forward.use_tensorwise_fp8_matmul = True
        sdnq_fp8_tw_tflops = benchmark_linear("SDNQ FP8 TW", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1)[0], x, steps)
        sdnq_fp8_tw_svd_tflops = benchmark_linear("SDNQ FP8 TW SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1, use_svd=True)[0], x, steps)

        sdnq.common.use_tensorwise_fp8_matmul = backup_tw_fp8
        sdnq.quantizer.use_tensorwise_fp8_matmul = backup_tw_fp8
        sdnq.forward.use_tensorwise_fp8_matmul = backup_tw_fp8
    else:
        print("Torch Compile is disabled, skipping quantized matmul tests.")

    sdnq_float_int16_tflops = benchmark_linear("SDNQ Float INT16", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int16", torch_dtype=dtype, use_quantized_matmul=False)[0], x, steps)
    sdnq_float_int12_tflops = benchmark_linear("SDNQ Float INT12", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int12", torch_dtype=dtype, use_quantized_matmul=False)[0], x, steps)
    sdnq_float_int8_tflops = benchmark_linear("SDNQ Float INT8", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int8", torch_dtype=dtype, use_quantized_matmul=False)[0], x, steps)
    sdnq_float_int4_tflops = benchmark_linear("SDNQ Float INT4", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int4", torch_dtype=dtype, use_quantized_matmul=False)[0], x, steps)

    sdnq_float_uint16_tflops = benchmark_linear("SDNQ Float UINT16", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint16", torch_dtype=dtype, use_quantized_matmul=False)[0], x, steps)
    sdnq_float_uint12_tflops = benchmark_linear("SDNQ Float UINT12", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint12", torch_dtype=dtype, use_quantized_matmul=False)[0], x, steps)
    sdnq_float_uint8_tflops = benchmark_linear("SDNQ Float UINT8", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint8", torch_dtype=dtype, use_quantized_matmul=False)[0], x, steps)
    sdnq_float_uint4_tflops = benchmark_linear("SDNQ Float UINT4", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint4", torch_dtype=dtype, use_quantized_matmul=False)[0], x, steps)

    sdnq_float_fp16_tflops = benchmark_linear("SDNQ Float FP16", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp16", torch_dtype=dtype, use_quantized_matmul=False)[0], x, steps)
    sdnq_float_fp12_tflops = benchmark_linear("SDNQ Float FP12", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp12", torch_dtype=dtype, use_quantized_matmul=False)[0], x, steps)
    sdnq_float_fp8_tflops = benchmark_linear("SDNQ Float FP8", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp8", torch_dtype=dtype, use_quantized_matmul=False)[0], x, steps)
    sdnq_float_fp4_tflops = benchmark_linear("SDNQ Float FP4", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp4", torch_dtype=dtype, use_quantized_matmul=False)[0], x, steps)

    sdnq_float_int16_svd_tflops = benchmark_linear("SDNQ Float INT16 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int16", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True)[0], x, steps)
    sdnq_float_int12_svd_tflops = benchmark_linear("SDNQ Float INT12 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int12", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True)[0], x, steps)
    sdnq_float_int8_svd_tflops = benchmark_linear("SDNQ Float INT8 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int8", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True)[0], x, steps)
    sdnq_float_int4_svd_tflops = benchmark_linear("SDNQ Float INT4 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="int4", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True)[0], x, steps)

    sdnq_float_uint16_svd_tflops = benchmark_linear("SDNQ Float UINT16 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint16", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True)[0], x, steps)
    sdnq_float_uint12_svd_tflops = benchmark_linear("SDNQ Float UINT12 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint12", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True)[0], x, steps)
    sdnq_float_uint8_svd_tflops = benchmark_linear("SDNQ Float UINT8 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint8", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True)[0], x, steps)
    sdnq_float_uint4_svd_tflops = benchmark_linear("SDNQ Float UINT4 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="uint4", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True)[0], x, steps)

    sdnq_float_fp16_svd_tflops = benchmark_linear("SDNQ Float FP16 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp16", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True)[0], x, steps)
    sdnq_float_fp12_svd_tflops = benchmark_linear("SDNQ Float FP12 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp12", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True)[0], x, steps)
    sdnq_float_fp8_svd_tflops = benchmark_linear("SDNQ Float FP8 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp8", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True)[0], x, steps)
    sdnq_float_fp4_svd_tflops = benchmark_linear("SDNQ Float FP4 SVD", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), weights_dtype="fp4", torch_dtype=dtype, use_quantized_matmul=False, use_svd=True)[0], x, steps)

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
    print("==================================================")
    if sdnq.common.use_torch_compile:
        print("SDNQ INT8 TFLOPS:", sdnq_int8_tflops)
        print("SDNQ FP8 TFLOPS:", sdnq_fp8_tflops)
        print("SDNQ FP8 TW TFLOPS:", sdnq_fp8_tw_tflops)
        print("SDNQ FP16 TFLOPS:", sdnq_fp16_tflops)
        print("==================================================")
        print("SDNQ INT8 SVD TFLOPS:", sdnq_int8_svd_tflops)
        print("SDNQ FP8 SVD TFLOPS:", sdnq_fp8_svd_tflops)
        print("SDNQ FP8 TW SVD TFLOPS:", sdnq_fp8_tw_svd_tflops)
        print("SDNQ FP16 SVD TFLOPS:", sdnq_fp16_svd_tflops)
        print("==================================================")
    print("SDNQ Float INT16 TFLOPS:", sdnq_float_int16_tflops)
    print("SDNQ Float INT12 TFLOPS:", sdnq_float_int12_tflops)
    print("SDNQ Float INT8 TFLOPS:", sdnq_float_int8_tflops)
    print("SDNQ Float INT4 TFLOPS:", sdnq_float_int4_tflops)
    print("==================================================")
    print("SDNQ Float UINT16 TFLOPS:", sdnq_float_uint16_tflops)
    print("SDNQ Float UINT12 TFLOPS:", sdnq_float_uint12_tflops)
    print("SDNQ Float UINT8 TFLOPS:", sdnq_float_uint8_tflops)
    print("SDNQ Float UINT4 TFLOPS:", sdnq_float_uint4_tflops)
    print("==================================================")
    print("SDNQ Float FP16 TFLOPS:", sdnq_float_fp16_tflops)
    print("SDNQ Float FP12 TFLOPS:", sdnq_float_fp12_tflops)
    print("SDNQ Float FP8 TFLOPS:", sdnq_float_fp8_tflops)
    print("SDNQ Float FP4 TFLOPS:", sdnq_float_fp4_tflops)
    print("==================================================")
    print("SDNQ Float INT16 SVD TFLOPS:", sdnq_float_int16_svd_tflops)
    print("SDNQ Float INT12 SVD TFLOPS:", sdnq_float_int12_svd_tflops)
    print("SDNQ Float INT8 SVD TFLOPS:", sdnq_float_int8_svd_tflops)
    print("SDNQ Float INT4 SVD TFLOPS:", sdnq_float_int4_svd_tflops)
    print("==================================================")
    print("SDNQ Float UINT16 SVD TFLOPS:", sdnq_float_uint16_svd_tflops)
    print("SDNQ Float UINT12 SVD TFLOPS:", sdnq_float_uint12_svd_tflops)
    print("SDNQ Float UINT8 SVD TFLOPS:", sdnq_float_uint8_svd_tflops)
    print("SDNQ Float UINT4 SVD TFLOPS:", sdnq_float_uint4_svd_tflops)
    print("==================================================")
    print("SDNQ Float FP16 SVD TFLOPS:", sdnq_float_fp16_svd_tflops)
    print("SDNQ Float FP12 SVD TFLOPS:", sdnq_float_fp12_svd_tflops)
    print("SDNQ Float FP8 SVD TFLOPS:", sdnq_float_fp8_svd_tflops)
    print("SDNQ Float FP4 SVD TFLOPS:", sdnq_float_fp4_svd_tflops)
    print("==================================================")
    print("")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a bucket list with a given dataset path")
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--mnk", default=8192, type=int)
    parser.add_argument("--dtype", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("-m", default=None, type=int)
    parser.add_argument("-n", default=None, type=int)
    parser.add_argument("-k", default=None, type=int)

    args = parser.parse_args()
    main(steps=args.steps, mnk=args.mnk, dtype=args.dtype, device=args.device, m=args.m, n=args.n, k=args.k)
