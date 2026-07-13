from collections.abc import Callable

import time
import platform

import torch
from tqdm import tqdm

import sdnq.common
import sdnq.kernel_wrappers
from sdnq.training import SDNQTensor
from sdnq.training.layers.linear.forward import quantized_linear_with_backward

from sdnq.training.layers.linear.linear_int8.linear_int8 import int8_matmul_with_backward
from sdnq.training.layers.linear.linear_int8.linear_int8_ckpt import int8_matmul_with_backward_ckpt
from sdnq.training.layers.linear.linear_int8.linear_int8_dynamic import int8_matmul_dynamic_with_backward
from sdnq.training.layers.linear.linear_int8.linear_int8_dynamic_ckpt import int8_matmul_dynamic_with_backward_ckpt

from sdnq.training.layers.linear.linear_uint8.linear_uint8 import uint8_matmul_with_backward
from sdnq.training.layers.linear.linear_uint8.linear_uint8_ckpt import uint8_matmul_with_backward_ckpt
from sdnq.training.layers.linear.linear_uint8.linear_uint8_dynamic import uint8_matmul_dynamic_with_backward
from sdnq.training.layers.linear.linear_uint8.linear_uint8_dynamic_ckpt import uint8_matmul_dynamic_with_backward_ckpt

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
    if sdnq.kernel_wrappers.use_openvino_mm:
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


def do_nothing(*args, **kwargs): # pylint: disable=unused-argument
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
        for _ in tqdm(range(steps)):
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
    sdnq_uint8_tflops = benchmark_linear("SDNQ UINT8", uint8_matmul_with_backward, x, SDNQTensor.from_float(y, weights_dtype="uint8", group_size=-1).requires_grad_(True), b, steps)
    sdnq_fp8_tflops = benchmark_linear("SDNQ FP8", fp8_matmul_with_backward, x, SDNQTensor.from_float(y, weights_dtype="fp8", group_size=-1).requires_grad_(True), b, steps)
    sdnq_fp16_tflops = benchmark_linear("SDNQ FP16", fp16_matmul_with_backward, x, SDNQTensor.from_float(y, weights_dtype="float16", group_size=-1).requires_grad_(True), b, steps)

    sdnq_float_uint16_tflops = benchmark_linear("SDNQ Float UINT16", quantized_linear_with_backward, x, SDNQTensor.from_float(y, weights_dtype="uint16").requires_grad_(True), b, steps)
    sdnq_float_int16_tflops = benchmark_linear("SDNQ Float INT16", quantized_linear_with_backward, x, SDNQTensor.from_float(y, weights_dtype="int16").requires_grad_(True), b, steps)
    sdnq_float_fp16_tflops = benchmark_linear("SDNQ Float FP16", quantized_linear_with_backward, x, SDNQTensor.from_float(y.clone(), weights_dtype="float16").requires_grad_(True), b, steps)
    sdnq_float_uint8_tflops = benchmark_linear("SDNQ Float UINT8", quantized_linear_with_backward, x, SDNQTensor.from_float(y, weights_dtype="uint8").requires_grad_(True), b, steps)
    sdnq_float_int8_tflops = benchmark_linear("SDNQ Float INT8", quantized_linear_with_backward, x, SDNQTensor.from_float(y, weights_dtype="int8").requires_grad_(True), b, steps)
    sdnq_float_fp8_tflops = benchmark_linear("SDNQ Float FP8", quantized_linear_with_backward, x, SDNQTensor.from_float(y, weights_dtype="fp8").requires_grad_(True), b, steps)

    sdnq_int8_dyn_float_tflops = benchmark_linear("SDNQ INT8 Dynamic Float", int8_matmul_dynamic_with_backward, x, y, b, steps)
    sdnq_int8_dyn_uint16_tflops = benchmark_linear("SDNQ INT8 Dynamic UINT16", int8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="uint16").requires_grad_(True), b, steps)
    sdnq_int8_dyn_int16_tflops = benchmark_linear("SDNQ INT8 Dynamic INT16", int8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="int16").requires_grad_(True), b, steps)
    sdnq_int8_dyn_fp16_tflops = benchmark_linear("SDNQ INT8 Dynamic FP16", int8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y.clone(), weights_dtype="float16").requires_grad_(True), b, steps)
    sdnq_int8_dyn_uint8_tflops = benchmark_linear("SDNQ INT8 Dynamic UINT8", int8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="uint8").requires_grad_(True), b, steps)
    sdnq_int8_dyn_int8_tflops = benchmark_linear("SDNQ INT8 Dynamic INT8", int8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="int8").requires_grad_(True), b, steps)
    sdnq_int8_dyn_fp8_tflops = benchmark_linear("SDNQ INT8 Dynamic FP8", int8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="fp8").requires_grad_(True), b, steps)

    sdnq_uint8_dyn_float_tflops = benchmark_linear("SDNQ UINT8 Dynamic Float", uint8_matmul_dynamic_with_backward, x, y, b, steps)
    sdnq_uint8_dyn_uint16_tflops = benchmark_linear("SDNQ UINT8 Dynamic UINT16", uint8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="uint16").requires_grad_(True), b, steps)
    sdnq_uint8_dyn_int16_tflops = benchmark_linear("SDNQ UINT8 Dynamic INT16", uint8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="int16").requires_grad_(True), b, steps)
    sdnq_uint8_dyn_fp16_tflops = benchmark_linear("SDNQ UINT8 Dynamic FP16", uint8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y.clone(), weights_dtype="float16").requires_grad_(True), b, steps)
    sdnq_uint8_dyn_uint8_tflops = benchmark_linear("SDNQ UINT8 Dynamic UINT8", uint8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="uint8").requires_grad_(True), b, steps)
    sdnq_uint8_dyn_int8_tflops = benchmark_linear("SDNQ UINT8 Dynamic INT8", uint8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="int8").requires_grad_(True), b, steps)
    sdnq_uint8_dyn_fp8_tflops = benchmark_linear("SDNQ UINT8 Dynamic FP8", uint8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="fp8").requires_grad_(True), b, steps)

    sdnq_fp8_dyn_float_tflops = benchmark_linear("SDNQ FP8 Dynamic Float", fp8_matmul_dynamic_with_backward, x, y, b, steps)
    sdnq_fp8_dyn_uint16_tflops = benchmark_linear("SDNQ FP8 Dynamic UINT16", fp8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="uint16").requires_grad_(True), b, steps)
    sdnq_fp8_dyn_int16_tflops = benchmark_linear("SDNQ FP8 Dynamic INT16", fp8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="int16").requires_grad_(True), b, steps)
    sdnq_fp8_dyn_fp16_tflops = benchmark_linear("SDNQ FP8 Dynamic FP16", fp8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y.clone(), weights_dtype="float16").requires_grad_(True), b, steps)
    sdnq_fp8_dyn_uint8_tflops = benchmark_linear("SDNQ FP8 Dynamic UINT8", fp8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="uint8").requires_grad_(True), b, steps)
    sdnq_fp8_dyn_int8_tflops = benchmark_linear("SDNQ FP8 Dynamic INT8", fp8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="int8").requires_grad_(True), b, steps)
    sdnq_fp8_dyn_fp8_tflops = benchmark_linear("SDNQ FP8 Dynamic FP8", fp8_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="fp8").requires_grad_(True), b, steps)

    sdnq_fp16_dyn_float_tflops = benchmark_linear("SDNQ FP16 Dynamic Float", fp16_matmul_dynamic_with_backward, x, y, b, steps)
    sdnq_fp16_dyn_uint16_tflops = benchmark_linear("SDNQ FP16 Dynamic UINT16", fp16_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="uint16").requires_grad_(True), b, steps)
    sdnq_fp16_dyn_int16_tflops = benchmark_linear("SDNQ FP16 Dynamic INT16", fp16_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="int16").requires_grad_(True), b, steps)
    sdnq_fp16_dyn_fp16_tflops = benchmark_linear("SDNQ FP16 Dynamic FP16", fp16_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y.clone(), weights_dtype="float16").requires_grad_(True), b, steps)
    sdnq_fp16_dyn_uint8_tflops = benchmark_linear("SDNQ FP16 Dynamic UINT8", fp16_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="uint8").requires_grad_(True), b, steps)
    sdnq_fp16_dyn_int8_tflops = benchmark_linear("SDNQ FP16 Dynamic INT8", fp16_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="int8").requires_grad_(True), b, steps)
    sdnq_fp16_dyn_fp8_tflops = benchmark_linear("SDNQ FP16 Dynamic FP8", fp16_matmul_dynamic_with_backward, x, SDNQTensor.from_float(y, weights_dtype="fp8").requires_grad_(True), b, steps)

    sdnq_int8_ckpt_tflops = benchmark_linear("SDNQ INT8 CKPT", int8_matmul_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="int8", group_size=-1).requires_grad_(True), b, steps)
    sdnq_uint8_ckpt_tflops = benchmark_linear("SDNQ UINT8 CKPT", uint8_matmul_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="uint8", group_size=-1).requires_grad_(True), b, steps)
    sdnq_fp8_ckpt_tflops = benchmark_linear("SDNQ FP8 CKPT", fp8_matmul_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="fp8", group_size=-1).requires_grad_(True), b, steps)
    sdnq_fp16_ckpt_tflops = benchmark_linear("SDNQ FP16 CKPT", fp16_matmul_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="fp8", group_size=-1).requires_grad_(True), b, steps)

    sdnq_int8_dyn_ckpt_float_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT Float", int8_matmul_dynamic_with_backward_ckpt, x, y, b, steps)
    sdnq_int8_dyn_ckpt_uint16_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT UINT16", int8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="uint16").requires_grad_(True), b, steps)
    sdnq_int8_dyn_ckpt_int16_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT INT16", int8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="int16").requires_grad_(True), b, steps)
    sdnq_int8_dyn_ckpt_fp16_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT FP16", int8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y.clone(), weights_dtype="float16").requires_grad_(True), b, steps)
    sdnq_int8_dyn_ckpt_uint8_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT UINT8", int8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="uint8").requires_grad_(True), b, steps)
    sdnq_int8_dyn_ckpt_int8_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT INT8", int8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="int8").requires_grad_(True), b, steps)
    sdnq_int8_dyn_ckpt_fp8_tflops = benchmark_linear("SDNQ INT8 Dynamic CKPT FP8", int8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="fp8").requires_grad_(True), b, steps)

    sdnq_uint8_dyn_ckpt_float_tflops = benchmark_linear("SDNQ UINT8 Dynamic CKPT Float", uint8_matmul_dynamic_with_backward_ckpt, x, y, b, steps)
    sdnq_uint8_dyn_ckpt_uint16_tflops = benchmark_linear("SDNQ UINT8 Dynamic CKPT UINT16", uint8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="uint16").requires_grad_(True), b, steps)
    sdnq_uint8_dyn_ckpt_int16_tflops = benchmark_linear("SDNQ UINT8 Dynamic CKPT INT16", uint8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="int16").requires_grad_(True), b, steps)
    sdnq_uint8_dyn_ckpt_fp16_tflops = benchmark_linear("SDNQ UINT8 Dynamic CKPT FP16", uint8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y.clone(), weights_dtype="float16").requires_grad_(True), b, steps)
    sdnq_uint8_dyn_ckpt_uint8_tflops = benchmark_linear("SDNQ UINT8 Dynamic CKPT UINT8", uint8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="uint8").requires_grad_(True), b, steps)
    sdnq_uint8_dyn_ckpt_int8_tflops = benchmark_linear("SDNQ UINT8 Dynamic CKPT INT8", uint8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="int8").requires_grad_(True), b, steps)
    sdnq_uint8_dyn_ckpt_fp8_tflops = benchmark_linear("SDNQ UINT8 Dynamic CKPT FP8", uint8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="fp8").requires_grad_(True), b, steps)

    sdnq_fp8_dyn_ckpt_float_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT Float", fp8_matmul_dynamic_with_backward_ckpt, x, y, b, steps)
    sdnq_fp8_dyn_ckpt_uint16_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT UINT16", fp8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="uint16").requires_grad_(True), b, steps)
    sdnq_fp8_dyn_ckpt_int16_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT INT16", fp8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="int16").requires_grad_(True), b, steps)
    sdnq_fp8_dyn_ckpt_fp16_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT FP16", fp8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y.clone(), weights_dtype="float16").requires_grad_(True), b, steps)
    sdnq_fp8_dyn_ckpt_uint8_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT UINT8", fp8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="uint8").requires_grad_(True), b, steps)
    sdnq_fp8_dyn_ckpt_int8_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT INT8", fp8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="int8").requires_grad_(True), b, steps)
    sdnq_fp8_dyn_ckpt_fp8_tflops = benchmark_linear("SDNQ FP8 Dynamic CKPT FP8", fp8_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="fp8").requires_grad_(True), b, steps)

    sdnq_fp16_dyn_ckpt_float_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT Float", fp16_matmul_dynamic_with_backward_ckpt, x, y, b, steps)
    sdnq_fp16_dyn_ckpt_uint16_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT UINT16", fp16_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="uint16").requires_grad_(True), b, steps)
    sdnq_fp16_dyn_ckpt_int16_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT INT16", fp16_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="int16").requires_grad_(True), b, steps)
    sdnq_fp16_dyn_ckpt_fp16_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT FP16", fp16_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y.clone(), weights_dtype="float16").requires_grad_(True), b, steps)
    sdnq_fp16_dyn_ckpt_uint8_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT UINT8", fp16_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="uint8").requires_grad_(True), b, steps)
    sdnq_fp16_dyn_ckpt_int8_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT INT8", fp16_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="int8").requires_grad_(True), b, steps)
    sdnq_fp16_dyn_ckpt_fp8_tflops = benchmark_linear("SDNQ FP16 Dynamic CKPT FP8", fp16_matmul_dynamic_with_backward_ckpt, x, SDNQTensor.from_float(y, weights_dtype="fp8").requires_grad_(True), b, steps)


    print("")
    print("==================================================")
    print("Platform:", platform.platform())
    print("Device:", get_device_name(device))
    print("Steps:", steps, "| MNK:", round((m*n*k)**(1/3)), "| Float:", dtype)
    print("M:", m, "| N:", n, "| K:", k)
    print("Torch Compile:", sdnq.common.use_torch_compile)
    print("Contiguous INT8 MM:", sdnq.kernel_wrappers.use_contiguous_int8_mm)
    print("Contiguous FP16 MM:", sdnq.kernel_wrappers.use_contiguous_fp16_mm)
    print("Tensorwise FP8 MM:", sdnq.kernel_wrappers.use_tensorwise_fp8_matmul)
    print("Triton MM:", sdnq.kernel_wrappers.use_triton_mm)
    print("OpenVINO MM:", sdnq.kernel_wrappers.use_openvino_mm)
    print("==================================================")
    print("PyTorch Float TFLOPS:", pytorch_float_tflops)
    print("SDNQ Float TFLOPS:", sdnq_float_tflops)
    print("==================================================")
    print("SDNQ INT8 TFLOPS:", sdnq_int8_tflops)
    print("SDNQ UINT8 TFLOPS:", sdnq_uint8_tflops)
    print("SDNQ FP8 TFLOPS:", sdnq_fp8_tflops)
    print("SDNQ FP16 TFLOPS:", sdnq_fp16_tflops)
    print("==================================================")
    print("SDNQ Float UINT16 TFLOPS:", sdnq_float_uint16_tflops)
    print("SDNQ Float INT16 TFLOPS:", sdnq_float_int16_tflops)
    print("SDNQ Float FP16 TFLOPS:", sdnq_float_fp16_tflops)
    print("SDNQ Float UINT8 TFLOPS:", sdnq_float_uint8_tflops)
    print("SDNQ Float INT8 TFLOPS:", sdnq_float_int8_tflops)
    print("SDNQ Float FP8 TFLOPS:", sdnq_float_fp8_tflops)
    print("==================================================")
    print("SDNQ INT8 Dynamic Float TFLOPS:", sdnq_int8_dyn_float_tflops)
    print("SDNQ INT8 Dynamic UINT16 TFLOPS:", sdnq_int8_dyn_uint16_tflops)
    print("SDNQ INT8 Dynamic INT16 TFLOPS:", sdnq_int8_dyn_int16_tflops)
    print("SDNQ INT8 Dynamic FP16 TFLOPS:", sdnq_int8_dyn_fp16_tflops)
    print("SDNQ INT8 Dynamic UINT8 TFLOPS:", sdnq_int8_dyn_uint8_tflops)
    print("SDNQ INT8 Dynamic INT8 TFLOPS:", sdnq_int8_dyn_int8_tflops)
    print("SDNQ INT8 Dynamic FP8 TFLOPS:", sdnq_int8_dyn_fp8_tflops)
    print("==================================================")
    print("SDNQ UINT8 Dynamic Float TFLOPS:", sdnq_uint8_dyn_float_tflops)
    print("SDNQ UINT8 Dynamic UINT16 TFLOPS:", sdnq_uint8_dyn_uint16_tflops)
    print("SDNQ UINT8 Dynamic INT16 TFLOPS:", sdnq_uint8_dyn_int16_tflops)
    print("SDNQ UINT8 Dynamic FP16 TFLOPS:", sdnq_uint8_dyn_fp16_tflops)
    print("SDNQ UINT8 Dynamic UINT8 TFLOPS:", sdnq_uint8_dyn_uint8_tflops)
    print("SDNQ UINT8 Dynamic INT8 TFLOPS:", sdnq_uint8_dyn_int8_tflops)
    print("SDNQ UINT8 Dynamic FP8 TFLOPS:", sdnq_uint8_dyn_fp8_tflops)
    print("==================================================")
    print("SDNQ FP8 Dynamic Float TFLOPS:", sdnq_fp8_dyn_float_tflops)
    print("SDNQ FP8 Dynamic UINT16 TFLOPS:", sdnq_fp8_dyn_uint16_tflops)
    print("SDNQ FP8 Dynamic INT16 TFLOPS:", sdnq_fp8_dyn_int16_tflops)
    print("SDNQ FP8 Dynamic FP16 TFLOPS:", sdnq_fp8_dyn_fp16_tflops)
    print("SDNQ FP8 Dynamic UINT8 TFLOPS:", sdnq_fp8_dyn_uint8_tflops)
    print("SDNQ FP8 Dynamic INT8 TFLOPS:", sdnq_fp8_dyn_int8_tflops)
    print("SDNQ FP8 Dynamic FP8 TFLOPS:", sdnq_fp8_dyn_fp8_tflops)
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
    print("SDNQ UINT8 CKPT TFLOPS:", sdnq_uint8_ckpt_tflops)
    print("SDNQ FP8 CKPT TFLOPS:", sdnq_fp8_ckpt_tflops)
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
    print("SDNQ UINT8 Dynamic CKPT Float TFLOPS:", sdnq_uint8_dyn_ckpt_float_tflops)
    print("SDNQ UINT8 Dynamic CKPT UINT16 TFLOPS:", sdnq_uint8_dyn_ckpt_uint16_tflops)
    print("SDNQ UINT8 Dynamic CKPT INT16 TFLOPS:", sdnq_uint8_dyn_ckpt_int16_tflops)
    print("SDNQ UINT8 Dynamic CKPT FP16 TFLOPS:", sdnq_uint8_dyn_ckpt_fp16_tflops)
    print("SDNQ UINT8 Dynamic CKPT UINT8 TFLOPS:", sdnq_uint8_dyn_ckpt_uint8_tflops)
    print("SDNQ UINT8 Dynamic CKPT INT8 TFLOPS:", sdnq_uint8_dyn_ckpt_int8_tflops)
    print("SDNQ UINT8 Dynamic CKPT FP8 TFLOPS:", sdnq_uint8_dyn_ckpt_fp8_tflops)
    print("==================================================")
    print("SDNQ FP8 Dynamic CKPT Float TFLOPS:", sdnq_fp8_dyn_ckpt_float_tflops)
    print("SDNQ FP8 Dynamic CKPT UINT16 TFLOPS:", sdnq_fp8_dyn_ckpt_uint16_tflops)
    print("SDNQ FP8 Dynamic CKPT INT16 TFLOPS:", sdnq_fp8_dyn_ckpt_int16_tflops)
    print("SDNQ FP8 Dynamic CKPT FP16 TFLOPS:", sdnq_fp8_dyn_ckpt_fp16_tflops)
    print("SDNQ FP8 Dynamic CKPT UINT8 TFLOPS:", sdnq_fp8_dyn_ckpt_uint8_tflops)
    print("SDNQ FP8 Dynamic CKPT INT8 TFLOPS:", sdnq_fp8_dyn_ckpt_int8_tflops)
    print("SDNQ FP8 Dynamic CKPT FP8 TFLOPS:", sdnq_fp8_dyn_ckpt_fp8_tflops)
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

    parser_args = parser.parse_args()
    main(steps=parser_args.steps, mnk=parser_args.mnk, dtype=parser_args.dtype, device=parser_args.device, m=parser_args.m, n=parser_args.n, k=parser_args.k)
