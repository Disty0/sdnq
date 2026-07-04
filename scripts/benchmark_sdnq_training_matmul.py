from collections.abc import Callable

import time
import torch
import platform
from tqdm import tqdm

import sdnq.common
from sdnq.training import SDNQTensor
from sdnq.training.layers.linear.forward import quantized_linear_with_backward

from sdnq.training.layers.linear.linear_int8.linear_int8 import int8_matmul_with_backward
from sdnq.training.layers.linear.linear_int8.linear_int8_ckpt import int8_matmul_with_backward_ckpt
from sdnq.training.layers.linear.linear_int8.linear_int8_dynamic import int8_matmul_dynamic_with_backward
from sdnq.training.layers.linear.linear_int8.linear_int8_dynamic_ckpt import int8_matmul_dynamic_with_backward_ckpt

from sdnq.training.layers.linear.linear_fp8_scaled.linear_fp8_scaled import fp8_scaled_matmul_with_backward
from sdnq.training.layers.linear.linear_fp8_scaled.linear_fp8_scaled_ckpt import fp8_scaled_matmul_with_backward_ckpt
from sdnq.training.layers.linear.linear_fp8_scaled.linear_fp8_scaled_dynamic import fp8_scaled_matmul_dynamic_with_backward
from sdnq.training.layers.linear.linear_fp8_scaled.linear_fp8_scaled_dynamic_ckpt import fp8_scaled_matmul_dynamic_with_backward_ckpt

from sdnq.training.layers.linear.linear_fp8.linear_fp8 import fp8_matmul_with_backward
from sdnq.training.layers.linear.linear_fp8.linear_fp8_ckpt import fp8_matmul_with_backward_ckpt
from sdnq.training.layers.linear.linear_fp8.linear_fp8_dynamic import fp8_matmul_dynamic_with_backward
from sdnq.training.layers.linear.linear_fp8.linear_fp8_dynamic_ckpt import fp8_matmul_dynamic_with_backward_ckpt

from sdnq.training.layers.linear.linear_fp16.linear_fp16 import fp16_matmul_with_backward
from sdnq.training.layers.linear.linear_fp16.linear_fp16_ckpt import fp16_matmul_with_backward_ckpt
from sdnq.training.layers.linear.linear_fp16.linear_fp16_dynamic import fp16_matmul_dynamic_with_backward
from sdnq.training.layers.linear.linear_fp16.linear_fp16_dynamic_ckpt import fp16_matmul_dynamic_with_backward_ckpt


def get_device_name(device: torch.device):
    device = torch.device(device)
    if sdnq.common.use_openvino_mm:
        from sdnq.kernels.openvino_mm import OV_DEVICE
        extra_device_str = OV_DEVICE + " of "
    else:
        extra_device_str = ""
    if device.type in {"xpu", "cuda"}:
        return extra_device_str + getattr(torch, device.type).get_device_name(device)
    try:
        import cpuinfo
        cpu_dict = cpuinfo.get_cpu_info()
        return extra_device_str + cpu_dict.get("arch", cpu_dict.get("arch_string_raw", platform.machine())) + " " + cpu_dict.get("hardware_raw", cpu_dict.get("brand_raw", "Unkwnown"))
    except Exception:
        return extra_device_str + platform.machine() + " " + (platform.processor() or "Unknown")
        

def do_nothing(*args, **kwargs):
    return


def get_sync_func(device: torch.device):
    device = torch.device(device)
    if device.type in {"xpu", "cuda"}:
        return getattr(torch, device.type).synchronize
    return do_nothing


def get_tflops(it_s: float, m: int, n: int, k: int) -> float:
    return round(it_s * ((3*2*m*k*n) + (2 * n * m)) / (10**12), 2)


def benchmark_linear(name: str, linear: Callable, x: torch.Tensor, y: torch.Tensor, b: torch.Tensor, steps: int):
    assert x.ndim == 2
    try:
        print(name)
        sync_func = get_sync_func(x.device)
        z = linear(x, y, b)
        loss = z.mean()
        loss.backward()
        sync_func(x.device)
        t0 = time.time()
        for i in tqdm(range(steps)):
            z = linear(x, y, b)
            loss = z.mean()
            loss.backward()
            sync_func(x.device)
        t1 = time.time()
        return get_tflops(steps/(t1 - t0), x.shape[0],z.shape[1],x.shape[1])
    except Exception:
        print(f"{name} test failed")
        return 0


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

    x = torch.randn(m,k, device=device, dtype=dtype).requires_grad_(True)
    y = torch.randn(n,k, device=device, dtype=dtype).requires_grad_(True)
    b = torch.randn(n, device=device, dtype=dtype).requires_grad_(True)

    pytorch_float_tflops = benchmark_linear("PyTorch Float", torch.nn.functional.linear, x, y, b, steps)
    sdnq_float_tflops = benchmark_linear("SDNQ Float", quantized_linear_with_backward, x, y, b, steps)

    sdnq_int8_tflops = benchmark_linear("SDNQ INT8", int8_matmul_with_backward, x, SDNQTensor.from_float(y, weights_dtype="int8", group_size=-1).requires_grad_(True), b, steps)
    sdnq_fp8_scaled_tflops = benchmark_linear("SDNQ FP8 Scaled", fp8_scaled_matmul_with_backward, x, SDNQTensor.from_float(y, weights_dtype="fp8", group_size=-1).requires_grad_(True), b, steps)
    sdnq_fp8_tflops = benchmark_linear("SDNQ FP8", fp8_matmul_with_backward, x, SDNQTensor.from_float(y, weights_dtype="fp8", group_size=-1).requires_grad_(True), b, steps)

    sdnq_int8_dyn_float_tflops = benchmark_linear("SDNQ INT8 Dynamic Float", int8_matmul_dynamic_with_backward, x, y, b, steps)
    sdnq_fp8_scaled_dyn_float_tflops = benchmark_linear("SDNQ FP8 Scaled Dynamic Float", fp8_scaled_matmul_dynamic_with_backward, x, y, b, steps)
    sdnq_fp8_dyn_float_tflops = benchmark_linear("SDNQ FP8 Dynamic Float", fp8_matmul_dynamic_with_backward, x, y, b, steps)

    sdnq_int8_ckpt_tflops = benchmark_linear("SDNQ INT8 CKPT", int8_matmul_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="int8", group_size=-1).requires_grad_(True), b, steps)
    sdnq_fp8_scaled_ckpt_tflops = benchmark_linear("SDNQ FP8 Scaled CKPT", fp8_scaled_matmul_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="fp8", group_size=-1).requires_grad_(True), b, steps)
    sdnq_fp8_ckpt_tflops = benchmark_linear("SDNQ FP8 CKPT", fp8_matmul_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="fp8", group_size=-1).requires_grad_(True), b, steps)

    sdnq_int8_dyn_ckpt_float_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT Float", int8_matmul_dynamic_with_backward_ckpt, x, y, b, steps)
    sdnq_fp8_scaled_dyn_ckpt_float_tflops = benchmark_linear("SDNQ FP8 Scaled Dynamic CKPT Float", fp8_scaled_matmul_dynamic_with_backward_ckpt, x, y, b, steps)
    sdnq_fp8_dyn_ckpt_float_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT Float", fp8_matmul_dynamic_with_backward_ckpt, x, y, b, steps)

    sdnq_fp16_tflops = benchmark_linear("SDNQ FP16", fp16_matmul_with_backward, x, SDNQTensor.from_float(y, weights_dtype="float16", group_size=-1).requires_grad_(True), b, steps)
    sdnq_fp16_dyn_float_tflops = benchmark_linear("SDNQ FP16 Dynamic Float", fp16_matmul_dynamic_with_backward, x, y, b, steps)
    sdnq_fp16_ckpt_tflops = benchmark_linear("SDNQ FP16 CKPT", fp16_matmul_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="fp8", group_size=-1).requires_grad_(True), b, steps)
    sdnq_fp16_dyn_ckpt_float_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT Float", fp16_matmul_dynamic_with_backward_ckpt, x, y, b, steps)

    print("")
    print("==================================================")
    print("Platform:", platform.platform())
    print("Device:", get_device_name(device))
    print("Steps:", steps, "| MNK:", round((m*n*k)**(1/3)), "| Float:", dtype)
    print("M:", m, "| N:", n, "| K:", k)
    print("Torch Compile:", sdnq.common.use_torch_compile)
    print("Contiguous MM:", sdnq.common.use_contiguous_mm)
    print("Triton MM:", sdnq.common.use_triton_mm)
    print("OpenVINO MM:", sdnq.common.use_openvino_mm)
    print("==================================================")
    print("PyTorch Float TFLOPS:", pytorch_float_tflops)
    print("SDNQ Float TFLOPS:", sdnq_float_tflops)
    print("==================================================")
    print("SDNQ INT8 TFLOPS:", sdnq_int8_tflops)
    print("SDNQ FP8 Scaled TFLOPS:", sdnq_fp8_scaled_tflops)
    print("SDNQ FP8 TFLOPS:", sdnq_fp8_tflops)
    print("SDNQ FP16 TFLOPS:", sdnq_fp16_tflops)
    print("==================================================")
    print("SDNQ INT8 Dynamic Float TFLOPS:", sdnq_int8_dyn_float_tflops)
    print("SDNQ FP8 Scaled Dynamic Float TFLOPS:", sdnq_fp8_scaled_dyn_float_tflops)
    print("SDNQ FP8 Dynamic Float TFLOPS:", sdnq_fp8_dyn_float_tflops)
    print("SDNQ FP16 Dynamic Float TFLOPS:", sdnq_fp16_dyn_float_tflops)
    print("==================================================")
    print("SDNQ INT8 CKPT TFLOPS:", sdnq_int8_ckpt_tflops)
    print("SDNQ FP8 Scaled CKPT TFLOPS:", sdnq_fp8_scaled_ckpt_tflops)
    print("SDNQ FP8 CKPT TFLOPS:", sdnq_fp8_ckpt_tflops)
    print("SDNQ FP16 CKPT TFLOPS:", sdnq_fp16_ckpt_tflops)
    print("==================================================")
    print("SDNQ INT8 Dynamic CKPT Float TFLOPS:", sdnq_int8_dyn_ckpt_float_tflops)
    print("SDNQ FP8 Scaled Dynamic CKPT Float TFLOPS:", sdnq_fp8_scaled_dyn_ckpt_float_tflops)
    print("SDNQ FP8 Dynamic CKPT Float TFLOPS:", sdnq_fp8_dyn_ckpt_float_tflops)
    print("SDNQ FP16 Dynamic CKPT Float TFLOPS:", sdnq_fp16_dyn_ckpt_float_tflops)
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
