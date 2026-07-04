import time
import platform

import torch
from tqdm import tqdm

import sdnq.common
import sdnq.quantizer
from sdnq import SDNQConfig, sdnq_quantize_layer


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


def do_nothing(*args, **kwargs): # pylint: disable=unused-argument
    return


def get_sync_func(device: torch.device):
    device = torch.device(device)
    if device.type in {"xpu", "cuda"}:
        return getattr(torch, device.type).synchronize
    return do_nothing


def get_tflops(it_s: float, m: int, n: int, k: int) -> float:
    return round(it_s * ((2*m*k*n) + (n * m)) / (10**12), 2)


def benchmark_linear(name: str, linear: torch.nn.Linear, x: torch.Tensor, steps: int):
    assert x.ndim == 2
    try:
        print(name)
        sync_func = get_sync_func(x.device)
        z = linear(x)
        sync_func(x.device)
        t0 = time.time()
        for _ in tqdm(range(steps)):
            z = linear(x)
            sync_func(x.device)
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
    sdnq_int8_tflops = benchmark_linear("SDNQ INT8", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), SDNQConfig(weights_dtype="int8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1))[0], x, steps)
    sdnq_int8_hadamard_tflops = benchmark_linear("SDNQ INT8 Hadamard", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), SDNQConfig(weights_dtype="int8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1, use_hadamard=True))[0], x, steps)

    sdnq_uint8_tflops = benchmark_linear("SDNQ UINT8", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), SDNQConfig(weights_dtype="uint8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1))[0], x, steps)
    sdnq_uint8_hadamard_tflops = benchmark_linear("SDNQ UINT8 Hadamard", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), SDNQConfig(weights_dtype="uint8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1, use_hadamard=True))[0], x, steps)

    sdnq_fp16_tflops = benchmark_linear("SDNQ FP16", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), SDNQConfig(weights_dtype="fp16", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1))[0], x, steps)
    sdnq_fp16_hadamard_tflops = benchmark_linear("SDNQ FP16 Hadamard", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), SDNQConfig(weights_dtype="fp16", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1, use_hadamard=True))[0], x, steps)

    backup_tw_fp8 = sdnq.common.use_tensorwise_fp8_matmul

    sdnq.common.use_tensorwise_fp8_matmul = False
    sdnq.quantizer.use_tensorwise_fp8_matmul = False
    sdnq.forward.use_tensorwise_fp8_matmul = False
    sdnq_fp8_scaled_tflops = benchmark_linear("SDNQ FP8 Scaled", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), SDNQConfig(weights_dtype="fp8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1))[0], x, steps)
    sdnq_fp8_scaled_hadamard_tflops = benchmark_linear("SDNQ FP8 Scaled Hadamard", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), SDNQConfig(weights_dtype="fp8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1, use_hadamard=True))[0], x, steps)

    sdnq.common.use_tensorwise_fp8_matmul = True
    sdnq.quantizer.use_tensorwise_fp8_matmul = True
    sdnq.forward.use_tensorwise_fp8_matmul = True
    sdnq_fp8_tflops = benchmark_linear("SDNQ FP8", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), SDNQConfig(weights_dtype="fp8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1))[0], x, steps)
    sdnq_fp8_hadamard_tflops = benchmark_linear("SDNQ FP8 Hadamard", sdnq_quantize_layer(torch.nn.Linear(k,n, bias=True).to(device, dtype=dtype), SDNQConfig(weights_dtype="fp8", torch_dtype=dtype, use_quantized_matmul=True, group_size=-1, use_hadamard=True))[0], x, steps)

    sdnq.common.use_tensorwise_fp8_matmul = backup_tw_fp8
    sdnq.quantizer.use_tensorwise_fp8_matmul = backup_tw_fp8
    sdnq.forward.use_tensorwise_fp8_matmul = backup_tw_fp8

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
    print("==================================================")
    print("SDNQ INT8 TFLOPS:", sdnq_int8_tflops)
    print("SDNQ UINT8 TFLOPS:", sdnq_uint8_tflops)
    print("SDNQ FP8 Scaled TFLOPS:", sdnq_fp8_scaled_tflops)
    print("SDNQ FP8 TFLOPS:", sdnq_fp8_tflops)
    print("SDNQ FP16 TFLOPS:", sdnq_fp16_tflops)
    print("==================================================")
    print("SDNQ INT8 Hadamard TFLOPS:", sdnq_int8_hadamard_tflops)
    print("SDNQ UINT8 Hadamard TFLOPS:", sdnq_uint8_hadamard_tflops)
    print("SDNQ FP8 Scaled Hadamard TFLOPS:", sdnq_fp8_scaled_hadamard_tflops)
    print("SDNQ FP8 Hadamard TFLOPS:", sdnq_fp8_hadamard_tflops)
    print("SDNQ FP16 Hadamard TFLOPS:", sdnq_fp16_hadamard_tflops)
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

    parser_args = parser.parse_args()
    main(steps=parser_args.steps, mnk=parser_args.mnk, dtype=parser_args.dtype, device=parser_args.device, m=parser_args.m, n=parser_args.n, k=parser_args.k)
