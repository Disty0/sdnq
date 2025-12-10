from typing import Optional, Union, Callable

import time
import torch
from tqdm import tqdm

import sdnq.common
from sdnq.training import SDNQTensor
from sdnq.training.layers.linear.forward import quantized_linear_with_backward

from sdnq.training.layers.linear.linear_int8 import int8_matmul_with_backward
from sdnq.training.layers.linear.linear_int8_ckpt import int8_matmul_with_backward_ckpt
from sdnq.training.layers.linear.linear_int8_dynamic import int8_matmul_dynamic_with_backward
from sdnq.training.layers.linear.linear_int8_dynamic_ckpt import int8_matmul_dynamic_with_backward_ckpt

from sdnq.training.layers.linear.linear_fp8 import fp8_matmul_with_backward
from sdnq.training.layers.linear.linear_fp8_ckpt import fp8_matmul_with_backward_ckpt
from sdnq.training.layers.linear.linear_fp8_dynamic import fp8_matmul_dynamic_with_backward
from sdnq.training.layers.linear.linear_fp8_dynamic_ckpt import fp8_matmul_dynamic_with_backward_ckpt

from sdnq.training.layers.linear.linear_fp8_tensorwise import fp8_matmul_tensorwise_with_backward
from sdnq.training.layers.linear.linear_fp8_tensorwise_ckpt import fp8_matmul_tensorwise_with_backward_ckpt
from sdnq.training.layers.linear.linear_fp8_tensorwise_dynamic import fp8_matmul_tensorwise_dynamic_with_backward
from sdnq.training.layers.linear.linear_fp8_tensorwise_dynamic_ckpt import fp8_matmul_tensorwise_dynamic_with_backward_ckpt

from sdnq.training.layers.linear.linear_fp16 import fp16_matmul_with_backward
from sdnq.training.layers.linear.linear_fp16_ckpt import fp16_matmul_with_backward_ckpt
from sdnq.training.layers.linear.linear_fp16_dynamic import fp16_matmul_dynamic_with_backward
from sdnq.training.layers.linear.linear_fp16_dynamic_ckpt import fp16_matmul_dynamic_with_backward_ckpt


def get_tflops(it_s: float, m: int, n: int, k: int) -> float:
    return round(it_s * ((3*2*m*k*n) + (2 * n * m)) / (10**12), 2)


def benchmark_linear(name: str, linear: Callable, x: torch.Tensor, y: torch.Tensor, b: torch.Tensor, steps: int):
    assert x.ndim == 2
    try:
        print(name)
        sync_func = getattr(torch, x.device.type).synchronize
        z = linear(x, y, b)
        loss = z.mean()
        loss.backward()
        sync_func()
        t0 = time.time()
        for i in tqdm(range(steps)):
            z = linear(x, y, b)
            loss = z.mean()
            loss.backward()
            sync_func()
        t1 = time.time()
        return get_tflops(steps/(t1 - t0), x.shape[0],z.shape[1],x.shape[1])
    except Exception:
        print(f"{name} test failed")
        return 0


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
    y = torch.randn(n,k, device=device, dtype=dtype)
    b = torch.randn(n, device=device, dtype=dtype)

    x.requires_grad_(True)
    y.requires_grad_(True)
    b.requires_grad_(True)

    yqf16 = SDNQTensor.from_float(y, weights_dtype="float16", group_size=-1)
    yqg16 = SDNQTensor.from_float(y, weights_dtype="int16", group_size=32)
    yqgu16 = SDNQTensor.from_float(y, weights_dtype="uint16", group_size=32)
    yqgf16 = SDNQTensor.from_float(y, weights_dtype="float16", group_size=32)
    yqf16.requires_grad_(True)
    yqg16.requires_grad_(True)
    yqgu16.requires_grad_(True)
    yqgf16.requires_grad_(True)

    yq = SDNQTensor.from_float(y, weights_dtype="int8", group_size=-1)
    yqg = SDNQTensor.from_float(y, weights_dtype="int8", group_size=32)
    yqgu = SDNQTensor.from_float(y, weights_dtype="uint8", group_size=32)
    yq.requires_grad_(True)
    yqg.requires_grad_(True)
    yqgu.requires_grad_(True)

    try:
        yqf = SDNQTensor.from_float(y, weights_dtype="fp8", group_size=-1)
        yqf.requires_grad_(True)
    except Exception:
        print("FP8 creation failed")
        yqf = None
    try:
        yqgf = SDNQTensor.from_float(y, weights_dtype="fp8", group_size=32)
        yqgf.requires_grad_(True)
    except Exception:
        print("Grouped FP8 creation failed")
        yqgf = None

    pytorch_float_tflops = benchmark_linear("PyTorch Float", torch.nn.functional.linear, x, y, b, steps)
    sdnq_float_tflops = benchmark_linear("SDNQ Float", quantized_linear_with_backward, x, y, b, steps)

    if sdnq.common.use_torch_compile:
        sdnq_int8_tflops = benchmark_linear("SDNQ INT8", int8_matmul_with_backward, x, yq, b, steps)
        sdnq_fp8_tflops = benchmark_linear("SDNQ FP8", fp8_matmul_with_backward, x, yqf, b, steps)
        sdnq_fp8_tw_tflops = benchmark_linear("SDNQ FP8 TW", fp8_matmul_tensorwise_with_backward, x, yqf, b, steps)
        sdnq_fp16_tflops = benchmark_linear("SDNQ FP16", fp16_matmul_with_backward, x, yqf16, b, steps)
    else:
        print("Torch Compile is disabled, skipping quantized matmul tests.")

    sdnq_float_uint16_tflops = benchmark_linear("SDNQ Float UINT16", quantized_linear_with_backward, x, yqgu16, b, steps)
    sdnq_float_int16_tflops = benchmark_linear("SDNQ Float INT16", quantized_linear_with_backward, x, yqg16, b, steps)
    sdnq_float_fp16_tflops = benchmark_linear("SDNQ Float FP16", quantized_linear_with_backward, x, yqgf16, b, steps)
    sdnq_float_uint8_tflops = benchmark_linear("SDNQ Float UINT8", quantized_linear_with_backward, x, yqgu, b, steps)
    sdnq_float_int8_tflops = benchmark_linear("SDNQ Float INT8", quantized_linear_with_backward, x, yqg, b, steps)
    sdnq_float_fp8_tflops = benchmark_linear("SDNQ Float FP8", quantized_linear_with_backward, x, yqgf, b, steps)

    if sdnq.common.use_torch_compile:
        sdnq_int8_dyn_float_tflops = benchmark_linear("SDNQ INT8 Dynamic Float", int8_matmul_dynamic_with_backward, x, y, b, steps)
        sdnq_int8_dyn_uint16_tflops = benchmark_linear("SDNQ INT8 Dynamic UINT16", int8_matmul_dynamic_with_backward, x, yqgu16, b, steps)
        sdnq_int8_dyn_int16_tflops = benchmark_linear("SDNQ INT8 Dynamic INT16", int8_matmul_dynamic_with_backward, x, yqg16, b, steps)
        sdnq_int8_dyn_fp16_tflops = benchmark_linear("SDNQ INT8 Dynamic FP16", int8_matmul_dynamic_with_backward, x, yqgf16, b, steps)
        sdnq_int8_dyn_uint8_tflops = benchmark_linear("SDNQ INT8 Dynamic UINT8", int8_matmul_dynamic_with_backward, x, yqgu, b, steps)
        sdnq_int8_dyn_int8_tflops = benchmark_linear("SDNQ INT8 Dynamic INT8", int8_matmul_dynamic_with_backward, x, yqg, b, steps)
        sdnq_int8_dyn_fp8_tflops = benchmark_linear("SDNQ INT8 Dynamic FP8", int8_matmul_dynamic_with_backward, x, yqgf, b, steps)

        sdnq_fp8_dyn_float_tflops = benchmark_linear("SDNQ FP8 Dynamic Float", fp8_matmul_dynamic_with_backward, x, y, b, steps)
        sdnq_fp8_dyn_uint16_tflops = benchmark_linear("SDNQ FP8 Dynamic UINT16", fp8_matmul_dynamic_with_backward, x, yqgu16, b, steps)
        sdnq_fp8_dyn_int16_tflops = benchmark_linear("SDNQ FP8 Dynamic INT16", fp8_matmul_dynamic_with_backward, x, yqg16, b, steps)
        sdnq_fp8_dyn_fp16_tflops = benchmark_linear("SDNQ FP8 Dynamic FP16", fp8_matmul_dynamic_with_backward, x, yqgf16, b, steps)
        sdnq_fp8_dyn_uint8_tflops = benchmark_linear("SDNQ FP8 Dynamic UINT8", fp8_matmul_dynamic_with_backward, x, yqgu, b, steps)
        sdnq_fp8_dyn_int8_tflops = benchmark_linear("SDNQ FP8 Dynamic INT8", fp8_matmul_dynamic_with_backward, x, yqg, b, steps)
        sdnq_fp8_dyn_fp8_tflops = benchmark_linear("SDNQ FP8 Dynamic FP8", fp8_matmul_dynamic_with_backward, x, yqgf, b, steps)

        sdnq_fp8_tw_dyn_float_tflops = benchmark_linear("SDNQ FP8 TW Dynamic Float", fp8_matmul_tensorwise_dynamic_with_backward, x, y, b, steps)
        sdnq_fp8_tw_dyn_uint16_tflops = benchmark_linear("SDNQ FP8 TW Dynamic UINT16", fp8_matmul_tensorwise_dynamic_with_backward, x, yqgu16, b, steps)
        sdnq_fp8_tw_dyn_int16_tflops = benchmark_linear("SDNQ FP8 TW Dynamic INT16", fp8_matmul_tensorwise_dynamic_with_backward, x, yqg16, b, steps)
        sdnq_fp8_tw_dyn_fp16_tflops = benchmark_linear("SDNQ FP8 TW Dynamic FP16", fp8_matmul_tensorwise_dynamic_with_backward, x, yqgf16, b, steps)
        sdnq_fp8_tw_dyn_uint8_tflops = benchmark_linear("SDNQ FP8 TW Dynamic UINT8", fp8_matmul_tensorwise_dynamic_with_backward, x, yqgu, b, steps)
        sdnq_fp8_tw_dyn_int8_tflops = benchmark_linear("SDNQ FP8 TW Dynamic INT8", fp8_matmul_tensorwise_dynamic_with_backward, x, yqg, b, steps)
        sdnq_fp8_tw_dyn_fp8_tflops = benchmark_linear("SDNQ FP8 TW Dynamic FP8", fp8_matmul_tensorwise_dynamic_with_backward, x, yqgf, b, steps)

        sdnq_fp16_dyn_float_tflops = benchmark_linear("SDNQ FP16 Dynamic Float", fp16_matmul_dynamic_with_backward, x, y, b, steps)
        sdnq_fp16_dyn_uint16_tflops = benchmark_linear("SDNQ FP16 Dynamic UINT16", fp16_matmul_dynamic_with_backward, x, yqgu16, b, steps)
        sdnq_fp16_dyn_int16_tflops = benchmark_linear("SDNQ FP16 Dynamic INT16", fp16_matmul_dynamic_with_backward, x, yqg16, b, steps)
        sdnq_fp16_dyn_fp16_tflops = benchmark_linear("SDNQ FP16 Dynamic FP16", fp16_matmul_dynamic_with_backward, x, yqgf16, b, steps)
        sdnq_fp16_dyn_uint8_tflops = benchmark_linear("SDNQ FP16 Dynamic UINT8", fp16_matmul_dynamic_with_backward, x, yqgu, b, steps)
        sdnq_fp16_dyn_int8_tflops = benchmark_linear("SDNQ FP16 Dynamic INT8", fp16_matmul_dynamic_with_backward, x, yqg, b, steps)
        sdnq_fp16_dyn_fp8_tflops = benchmark_linear("SDNQ FP16 Dynamic FP8", fp16_matmul_dynamic_with_backward, x, yqgf, b, steps)

        sdnq_int8_ckpt_tflops = benchmark_linear("SDNQ INT8 CKPT", int8_matmul_with_backward_ckpt, x, yq, b, steps)
        sdnq_fp8_ckpt_tflops = benchmark_linear("SDNQ FP8 CKPT", fp8_matmul_with_backward_ckpt, x, yqf, b, steps)
        sdnq_fp8_tw_ckpt_tflops = benchmark_linear("SDNQ FP8 TW CKPT", fp8_matmul_tensorwise_with_backward_ckpt, x, yqf, b, steps)
        sdnq_fp16_ckpt_tflops = benchmark_linear("SDNQ FP16 CKPT", fp16_matmul_with_backward_ckpt, x, yqf, b, steps)

        sdnq_int8_dyn_ckpt_float_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT Float", int8_matmul_dynamic_with_backward_ckpt, x, y, b, steps)
        sdnq_int8_dyn_ckpt_uint16_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT UINT16", int8_matmul_dynamic_with_backward_ckpt, x, yqgu16, b, steps)
        sdnq_int8_dyn_ckpt_int16_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT INT16", int8_matmul_dynamic_with_backward_ckpt, x, yqg16, b, steps)
        sdnq_int8_dyn_ckpt_fp16_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT FP16", int8_matmul_dynamic_with_backward_ckpt, x, yqgf16, b, steps)
        sdnq_int8_dyn_ckpt_uint8_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT UINT8", int8_matmul_dynamic_with_backward_ckpt, x, yqgu, b, steps)
        sdnq_int8_dyn_ckpt_int8_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT INT8", int8_matmul_dynamic_with_backward_ckpt, x, yqg, b, steps)
        sdnq_int8_dyn_ckpt_fp8_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT FP8", int8_matmul_dynamic_with_backward_ckpt, x, yqgf, b, steps)

        sdnq_fp8_dyn_ckpt_float_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT Float", fp8_matmul_dynamic_with_backward_ckpt, x, y, b, steps)
        sdnq_fp8_dyn_ckpt_uint16_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT UINT16", fp8_matmul_dynamic_with_backward_ckpt, x, yqgu16, b, steps)
        sdnq_fp8_dyn_ckpt_int16_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT INT16", fp8_matmul_dynamic_with_backward_ckpt, x, yqg16, b, steps)
        sdnq_fp8_dyn_ckpt_fp16_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT FP16", fp8_matmul_dynamic_with_backward_ckpt, x, yqgf16, b, steps)
        sdnq_fp8_dyn_ckpt_uint8_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT UINT8", fp8_matmul_dynamic_with_backward_ckpt, x, yqgu, b, steps)
        sdnq_fp8_dyn_ckpt_int8_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT INT8", fp8_matmul_dynamic_with_backward_ckpt, x, yqg, b, steps)
        sdnq_fp8_dyn_ckpt_fp8_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT FP8", fp8_matmul_dynamic_with_backward_ckpt, x, yqgf, b, steps)

        sdnq_fp8_tw_dyn_ckpt_float_tflops = benchmark_linear("SDNQ FP8 TW Dynamic CKPT Float", fp8_matmul_tensorwise_dynamic_with_backward_ckpt, x, y, b, steps)
        sdnq_fp8_tw_dyn_ckpt_uint16_tflops = benchmark_linear("SDNQ FP8 TW Dynamic CKPT UINT16", fp8_matmul_tensorwise_dynamic_with_backward_ckpt, x, yqgu16, b, steps)
        sdnq_fp8_tw_dyn_ckpt_int16_tflops = benchmark_linear("SDNQ FP8 TW Dynamic CKPT INT16", fp8_matmul_tensorwise_dynamic_with_backward_ckpt, x, yqg16, b, steps)
        sdnq_fp8_tw_dyn_ckpt_fp16_tflops = benchmark_linear("SDNQ FP8 TW Dynamic CKPT FP16", fp8_matmul_tensorwise_dynamic_with_backward_ckpt, x, yqgf16, b, steps)
        sdnq_fp8_tw_dyn_ckpt_uint8_tflops = benchmark_linear("SDNQ FP8 TW Dynamic CKPT UINT8", fp8_matmul_tensorwise_dynamic_with_backward_ckpt, x, yqgu, b, steps)
        sdnq_fp8_tw_dyn_ckpt_int8_tflops = benchmark_linear("SDNQ FP8 TW Dynamic CKPT INT8", fp8_matmul_tensorwise_dynamic_with_backward_ckpt, x, yqg, b, steps)
        sdnq_fp8_tw_dyn_ckpt_fp8_tflops = benchmark_linear("SDNQ FP8 TW Dynamic CKPT FP8", fp8_matmul_tensorwise_dynamic_with_backward_ckpt, x, yqgf, b, steps)

        sdnq_fp16_dyn_ckpt_float_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT Float", fp16_matmul_dynamic_with_backward_ckpt, x, y, b, steps)
        sdnq_fp16_dyn_ckpt_uint16_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT UINT16", fp16_matmul_dynamic_with_backward_ckpt, x, yqgu16, b, steps)
        sdnq_fp16_dyn_ckpt_int16_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT INT16", fp16_matmul_dynamic_with_backward_ckpt, x, yqg16, b, steps)
        sdnq_fp16_dyn_ckpt_fp16_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT FP16", fp16_matmul_dynamic_with_backward_ckpt, x, yqgf16, b, steps)
        sdnq_fp16_dyn_ckpt_uint8_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT UINT8", fp16_matmul_dynamic_with_backward_ckpt, x, yqgu, b, steps)
        sdnq_fp16_dyn_ckpt_int8_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT INT8", fp16_matmul_dynamic_with_backward_ckpt, x, yqg, b, steps)
        sdnq_fp16_dyn_ckpt_fp8_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT FP8", fp16_matmul_dynamic_with_backward_ckpt, x, yqgf, b, steps)
    else:
        print("Torch Compile is disabled, skipping quantized matmul tests.")


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
    print("SDNQ Float TFLOPS:", sdnq_float_tflops)
    if sdnq.common.use_torch_compile:
        print("==================================================")
        print("SDNQ INT8 TFLOPS:", sdnq_int8_tflops)
        print("SDNQ FP8 TFLOPS:", sdnq_fp8_tflops)
        print("SDNQ FP8 TW TFLOPS:", sdnq_fp8_tw_tflops)
        print("SDNQ FP16 TFLOPS:", sdnq_fp16_tflops)
    print("==================================================")
    print("SDNQ Float UINT16 TFLOPS:", sdnq_float_uint16_tflops)
    print("SDNQ Float INT16 TFLOPS:", sdnq_float_int16_tflops)
    print("SDNQ Float FP16 TFLOPS:", sdnq_float_fp16_tflops)
    print("SDNQ Float UINT8 TFLOPS:", sdnq_float_uint8_tflops)
    print("SDNQ Float INT8 TFLOPS:", sdnq_float_int8_tflops)
    print("SDNQ Float FP8 TFLOPS:", sdnq_float_fp8_tflops)
    if sdnq.common.use_torch_compile:
        print("==================================================")
        print("SDNQ INT8 Dynamic Float TFLOPS:", sdnq_int8_dyn_float_tflops)
        print("SDNQ INT8 Dynamic UINT16 TFLOPS:", sdnq_int8_dyn_uint16_tflops)
        print("SDNQ INT8 Dynamic INT16 TFLOPS:", sdnq_int8_dyn_int16_tflops)
        print("SDNQ INT8 Dynamic FP16 TFLOPS:", sdnq_int8_dyn_fp16_tflops)
        print("SDNQ INT8 Dynamic UINT8 TFLOPS:", sdnq_int8_dyn_uint8_tflops)
        print("SDNQ INT8 Dynamic INT8 TFLOPS:", sdnq_int8_dyn_int8_tflops)
        print("SDNQ INT8 Dynamic FP8 TFLOPS:", sdnq_int8_dyn_fp8_tflops)
        print("==================================================")
        print("SDNQ FP8 Dynamic Float TFLOPS:", sdnq_fp8_dyn_float_tflops)
        print("SDNQ FP8 Dynamic UINT16 TFLOPS:", sdnq_fp8_dyn_uint16_tflops)
        print("SDNQ FP8 Dynamic INT16 TFLOPS:", sdnq_fp8_dyn_int16_tflops)
        print("SDNQ FP8 Dynamic FP16 TFLOPS:", sdnq_fp8_dyn_fp16_tflops)
        print("SDNQ FP8 Dynamic UINT8 TFLOPS:", sdnq_fp8_dyn_uint8_tflops)
        print("SDNQ FP8 Dynamic INT8 TFLOPS:", sdnq_fp8_dyn_int8_tflops)
        print("SDNQ FP8 Dynamic FP8 TFLOPS:", sdnq_fp8_dyn_fp8_tflops)
        print("==================================================")
        print("SDNQ FP8 TW Dynamic Float TFLOPS:", sdnq_fp8_tw_dyn_float_tflops)
        print("SDNQ FP8 TW Dynamic UINT16 TFLOPS:", sdnq_fp8_tw_dyn_uint16_tflops)
        print("SDNQ FP8 TW Dynamic INT16 TFLOPS:", sdnq_fp8_tw_dyn_int16_tflops)
        print("SDNQ FP8 TW Dynamic FP16 TFLOPS:", sdnq_fp8_tw_dyn_fp16_tflops)
        print("SDNQ FP8 TW Dynamic UINT8 TFLOPS:", sdnq_fp8_tw_dyn_uint8_tflops)
        print("SDNQ FP8 TW Dynamic INT8 TFLOPS:", sdnq_fp8_tw_dyn_int8_tflops)
        print("SDNQ FP8 TW Dynamic FP8 TFLOPS:", sdnq_fp8_tw_dyn_fp8_tflops)
        print("==================================================")
        print("SDNQ FP16 Dynamic Float TFLOPS:", sdnq_fp16_dyn_float_tflops)
        print("SDNQ FP16 Dynamic UINT16 TFLOPS:", sdnq_fp16_dyn_uint16_tflops)
        print("SDNQ FP16 Dynamic INT16 TFLOPS:", sdnq_fp16_dyn_int16_tflops)
        print("SDNQ FP16 Dynamic FP16 TFLOPS:", sdnq_fp16_dyn_fp16_tflops)
        print("SDNQ FP16 Dynamic UINT8 TFLOPS:", sdnq_fp16_dyn_uint8_tflops)
        print("SDNQ FP16 Dynamic INT8 TFLOPS:", sdnq_fp16_dyn_int8_tflops)
        print("SDNQ FP16 Dynamic FP8 TFLOPS:", sdnq_fp16_dyn_fp8_tflops)
        print("==================================================")
        print("SDNQ INT8 CKPT TFLOPS:", sdnq_int8_ckpt_tflops)
        print("SDNQ FP8 CKPT TFLOPS:", sdnq_fp8_ckpt_tflops)
        print("SDNQ FP8 TW CKPT TFLOPS:", sdnq_fp8_tw_ckpt_tflops)
        print("SDNQ FP16 CKPT TFLOPS:", sdnq_fp16_ckpt_tflops)
        print("==================================================")
        print("SDNQ INT8 Dynamic CKPT Float TFLOPS:", sdnq_int8_dyn_ckpt_float_tflops)
        print("SDNQ INT8 Dynamic CKPT UINT16 TFLOPS:", sdnq_int8_dyn_ckpt_uint16_tflops)
        print("SDNQ INT8 Dynamic CKPT INT16 TFLOPS:", sdnq_int8_dyn_ckpt_int16_tflops)
        print("SDNQ INT8 Dynamic CKPT FP16 TFLOPS:", sdnq_int8_dyn_ckpt_fp16_tflops)
        print("SDNQ INT8 Dynamic CKPT UINT8 TFLOPS:", sdnq_int8_dyn_ckpt_uint8_tflops)
        print("SDNQ INT8 Dynamic CKPT INT8 TFLOPS:", sdnq_int8_dyn_ckpt_int8_tflops)
        print("SDNQ INT8 Dynamic CKPT FP8 TFLOPS:", sdnq_int8_dyn_ckpt_fp8_tflops)
        print("==================================================")
        print("SDNQ FP8 Dynamic CKPT Float TFLOPS:", sdnq_fp8_dyn_ckpt_float_tflops)
        print("SDNQ FP8 Dynamic CKPT UINT16 TFLOPS:", sdnq_fp8_dyn_ckpt_uint16_tflops)
        print("SDNQ FP8 Dynamic CKPT INT16 TFLOPS:", sdnq_fp8_dyn_ckpt_int16_tflops)
        print("SDNQ FP8 Dynamic CKPT FP16 TFLOPS:", sdnq_fp8_dyn_ckpt_fp16_tflops)
        print("SDNQ FP8 Dynamic CKPT UINT8 TFLOPS:", sdnq_fp8_dyn_ckpt_uint8_tflops)
        print("SDNQ FP8 Dynamic CKPT INT8 TFLOPS:", sdnq_fp8_dyn_ckpt_int8_tflops)
        print("SDNQ FP8 Dynamic CKPT FP8 TFLOPS:", sdnq_fp8_dyn_ckpt_fp8_tflops)
        print("==================================================")
        print("SDNQ FP8 TW Dynamic CKPT Float TFLOPS:", sdnq_fp8_tw_dyn_ckpt_float_tflops)
        print("SDNQ FP8 TW Dynamic CKPT UINT16 TFLOPS:", sdnq_fp8_tw_dyn_ckpt_uint16_tflops)
        print("SDNQ FP8 TW Dynamic CKPT INT16 TFLOPS:", sdnq_fp8_tw_dyn_ckpt_int16_tflops)
        print("SDNQ FP8 TW Dynamic CKPT FP16 TFLOPS:", sdnq_fp8_tw_dyn_ckpt_fp16_tflops)
        print("SDNQ FP8 TW Dynamic CKPT UINT8 TFLOPS:", sdnq_fp8_tw_dyn_ckpt_uint8_tflops)
        print("SDNQ FP8 TW Dynamic CKPT INT8 TFLOPS:", sdnq_fp8_tw_dyn_ckpt_int8_tflops)
        print("SDNQ FP8 TW Dynamic CKPT FP8 TFLOPS:", sdnq_fp8_tw_dyn_ckpt_fp8_tflops)
        print("==================================================")
        print("SDNQ FP16 Dynamic CKPT Float TFLOPS:", sdnq_fp16_dyn_ckpt_float_tflops)
        print("SDNQ FP16 Dynamic CKPT UINT16 TFLOPS:", sdnq_fp16_dyn_ckpt_uint16_tflops)
        print("SDNQ FP16 Dynamic CKPT INT16 TFLOPS:", sdnq_fp16_dyn_ckpt_int16_tflops)
        print("SDNQ FP16 Dynamic CKPT FP16 TFLOPS:", sdnq_fp16_dyn_ckpt_fp16_tflops)
        print("SDNQ FP16 Dynamic CKPT UINT8 TFLOPS:", sdnq_fp16_dyn_ckpt_uint8_tflops)
        print("SDNQ FP16 Dynamic CKPT INT8 TFLOPS:", sdnq_fp16_dyn_ckpt_int8_tflops)
        print("SDNQ FP16 Dynamic CKPT FP8 TFLOPS:", sdnq_fp16_dyn_ckpt_fp8_tflops)
    print("==================================================")
    print("")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create a bucket list with a given dataset path")
    parser.add_argument("--steps", default=50, type=int)
    parser.add_argument("--mnk", default=8192, type=int)
    parser.add_argument("--dtype", default=None, type=str)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("-m", default=None, type=int)
    parser.add_argument("-n", default=None, type=int)
    parser.add_argument("-k", default=None, type=int)

    args = parser.parse_args()
    main(steps=args.steps, mnk=args.mnk, dtype=args.dtype, device=args.device, m=args.m, n=args.n, k=args.k)
