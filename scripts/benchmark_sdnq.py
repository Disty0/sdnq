import time
import torch
from tqdm import tqdm

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

def get_tflops(it_s, m, n, k):
    return round(it_s * ((3*2*m*k*n) + (2 * n * m)) / (10**12), 2)


steps = 50
device = "cuda" if torch.cuda.is_available() else "xpu"
sync_func = getattr(torch, torch.device(device).type).synchronize
dtype = torch.bfloat16

m = 4*4096
k = 4096
n = 4*2048

x = torch.randn(m,k, device=device, dtype=dtype)
y = torch.randn(n,k, device=device, dtype=dtype)
b = torch.randn(n, device=device, dtype=dtype)

x.requires_grad_(True)
y.requires_grad_(True)
b.requires_grad_(True)

yq = SDNQTensor.from_float(y, qtype="int8", group_size=-1)
yqg = SDNQTensor.from_float(y, qtype="int8", group_size=32)
yqgu = SDNQTensor.from_float(y, qtype="uint8", group_size=32)

yq.requires_grad_(True)
yqg.requires_grad_(True)
yqgu.requires_grad_(True)

try:
    yqf = SDNQTensor.from_float(y, qtype="fp8", group_size=-1)
    yqf.requires_grad_(True)
except Exception:
    print("FP8 creation failed")
try:
    yqgf = SDNQTensor.from_float(y, qtype="fp8", group_size=32)
    yqgf.requires_grad_(True)
except Exception:
    print("Grouped FP8 creation failed")

    
try:
    print("PyTorch Float:")
    z = torch.nn.functional.linear(x, y, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = torch.nn.functional.linear(x, y, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    pytorch_float_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("PyTorch Float test failed")
    pytorch_float_tflops = 0


try:
    print("SDNQ Float:")
    z = quantized_linear_with_backward(x, y, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = quantized_linear_with_backward(x, y, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_float_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ Float test failed")
    sdnq_float_uint8_tflops = 0


try:
    print("SDNQ INT8:")
    z = int8_matmul_with_backward(x, yq, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = int8_matmul_with_backward(x, yq, b)
        loss = z.mean()
        loss.backward()
        g_int8 = x.grad.clone()
        sync_func()
    t1 = time.time()
    sdnq_int8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ INT8 test failed")
    sdnq_int8_tflops = 0

try:
    print("SDNQ FP8:")
    z = fp8_matmul_with_backward(x, yqf, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_with_backward(x, yqf, b)
        loss = z.mean()
        loss.backward()
        g_int8 = x.grad.clone()
        sync_func()
    t1 = time.time()
    sdnq_fp8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 test failed")
    sdnq_fp8_tflops = 0


try:
    print("SDNQ FP8 TW:")
    z = fp8_matmul_tensorwise_with_backward(x, yqf, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_tensorwise_with_backward(x, yqf, b)
        loss = z.mean()
        loss.backward()
        g_int8 = x.grad.clone()
        sync_func()
    t1 = time.time()
    sdnq_fp8_tw_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 TW test failed")
    sdnq_fp8_tw_tflops = 0


try:
    print("SDNQ Float UINT8:")
    z = quantized_linear_with_backward(x, yqgu, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = quantized_linear_with_backward(x, yqgu, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_float_uint8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ Float UINT8 test failed")
    sdnq_float_uint8_tflops = 0


try:
    print("SDNQ Float INT8:")
    z = quantized_linear_with_backward(x, yqg, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = quantized_linear_with_backward(x, yqg, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_float_int8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ Float INT8 test failed")
    sdnq_float_int8_tflops = 0


try:
    print("SDNQ Float FP8:")
    z = quantized_linear_with_backward(x, yqgf, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = quantized_linear_with_backward(x, yqgf, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_float_fp8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ Float FP8 test failed")
    sdnq_float_fp8_tflops = 0


try:
    print("SDNQ INT8 Dynamic Float:")
    z = int8_matmul_dynamic_with_backward(x, y, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = int8_matmul_dynamic_with_backward(x, y, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_int8_dyn_float_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ INT8 Dynamic Float test failed")
    sdnq_int8_dyn_float_tflops = 0


try:
    print("SDNQ INT8 Dynamic UINT8:")
    z = int8_matmul_dynamic_with_backward(x, yqgu, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = int8_matmul_dynamic_with_backward(x, yqgu, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_int8_dyn_uint8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ INT8 Dynamic UINT8 test failed")
    sdnq_int8_dyn_uint8_tflops = 0


try:
    print("SDNQ INT8 Dynamic INT8:")
    z = int8_matmul_dynamic_with_backward(x, yqg, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = int8_matmul_dynamic_with_backward(x, yqg, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_int8_dyn_int8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ INT8 Dynamic INT8 test failed")
    sdnq_int8_dyn_int8_tflops = 0


try:
    print("SDNQ INT8 Dynamic FP8:")
    z = int8_matmul_dynamic_with_backward(x, yqgf, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = int8_matmul_dynamic_with_backward(x, yqgf, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_int8_dyn_fp8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ INT8 Dynamic FP8 test failed")
    sdnq_int8_dyn_fp8_tflops = 0

try:
    print("SDNQ FP8 Dynamic Float:")
    z = fp8_matmul_dynamic_with_backward(x, y, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_dynamic_with_backward(x, y, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_dyn_float_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 Dynamic Float test failed")
    sdnq_fp8_dyn_float_tflops = 0


try:
    print("SDNQ FP8 Dynamic UINT8:")
    z = fp8_matmul_dynamic_with_backward(x, yqgu, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_dynamic_with_backward(x, yqgu, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_dyn_uint8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 Dynamic UINT8 test failed")
    sdnq_fp8_dyn_uint8_tflops = 0


try:
    print("SDNQ FP8 Dynamic INT8:")
    z = fp8_matmul_dynamic_with_backward(x, yqg, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_dynamic_with_backward(x, yqg, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_dyn_int8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 Dynamic INT8 test failed")
    sdnq_fp8_dyn_int8_tflops = 0


try:
    print("SDNQ FP8 Dynamic FP8:")
    z = fp8_matmul_dynamic_with_backward(x, yqgf, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_dynamic_with_backward(x, yqgf, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_dyn_fp8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 Dynamic FP8 test failed")
    sdnq_fp8_dyn_fp8_tflops = 0


try:
    print("SDNQ INT8 Dynamic FP8:")
    z = int8_matmul_dynamic_with_backward(x, yqgf, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = int8_matmul_dynamic_with_backward(x, yqgf, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_int8_dyn_fp8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ INT8 Dynamic FP8 test failed")
    sdnq_int8_dyn_fp8_tflops = 0

try:
    print("SDNQ FP8 TW Dynamic Float:")
    z = fp8_matmul_tensorwise_dynamic_with_backward(x, y, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_tensorwise_dynamic_with_backward(x, y, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_tw_dyn_float_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 TW Dynamic Float test failed")
    sdnq_fp8_tw_dyn_float_tflops = 0


try:
    print("SDNQ FP8 TW Dynamic UINT8:")
    z = fp8_matmul_tensorwise_dynamic_with_backward(x, yqgu, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_tensorwise_dynamic_with_backward(x, yqgu, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_tw_dyn_uint8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 TW Dynamic UINT8 test failed")
    sdnq_fp8_tw_dyn_uint8_tflops = 0


try:
    print("SDNQ FP8 TW Dynamic INT8:")
    z = fp8_matmul_tensorwise_dynamic_with_backward(x, yqg, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_tensorwise_dynamic_with_backward(x, yqg, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_tw_dyn_int8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 TW Dynamic INT8 test failed")
    sdnq_fp8_tw_dyn_int8_tflops = 0


try:
    print("SDNQ FP8 TW Dynamic FP8:")
    z = fp8_matmul_tensorwise_dynamic_with_backward(x, yqgf, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_tensorwise_dynamic_with_backward(x, yqgf, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_tw_dyn_fp8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 TW Dynamic FP8 test failed")
    sdnq_fp8_tw_dyn_fp8_tflops = 0


try:
    print("SDNQ INT8 CKPT:")
    z = int8_matmul_with_backward_ckpt(x, yq, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = int8_matmul_with_backward_ckpt(x, yq, b)
        loss = z.mean()
        loss.backward()
        g_int8 = x.grad.clone()
        sync_func()
    t1 = time.time()
    sdnq_int8_ckpt_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ INT8 CKPT test failed")
    sdnq_int8_ckpt_tflops = 0

try:
    print("SDNQ FP8 CKPT:")
    z = fp8_matmul_with_backward_ckpt(x, yqf, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_with_backward_ckpt(x, yqf, b)
        loss = z.mean()
        loss.backward()
        g_int8 = x.grad.clone()
        sync_func()
    t1 = time.time()
    sdnq_fp8_ckpt_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 CKPT test failed")
    sdnq_fp8_ckpt_tflops = 0


try:
    print("SDNQ FP8 TW CKPT:")
    z = fp8_matmul_tensorwise_with_backward_ckpt(x, yqf, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_tensorwise_with_backward_ckpt(x, yqf, b)
        loss = z.mean()
        loss.backward()
        g_int8 = x.grad.clone()
        sync_func()
    t1 = time.time()
    sdnq_fp8_tw_ckpt_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 TW CKPT test failed")
    sdnq_fp8_tw_ckpt_tflops = 0


try:
    print("SDNQ INT8 Dynamic Float:")
    z = int8_matmul_dynamic_with_backward_ckpt(x, y, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = int8_matmul_dynamic_with_backward_ckpt(x, y, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_int8_dyn_ckpt_float_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ INT8 Dynamic CKPT Float test failed")
    sdnq_int8_dyn_ckpt_float_tflops = 0


try:
    print("SDNQ INT8 Dynamic CKPT UINT8:")
    z = int8_matmul_dynamic_with_backward_ckpt(x, yqgu, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = int8_matmul_dynamic_with_backward_ckpt(x, yqgu, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_int8_dyn_ckpt_uint8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ INT8 Dynamic CKPT UINT8 test failed")
    sdnq_int8_dyn_ckpt_uint8_tflops = 0


try:
    print("SDNQ INT8 Dynamic CKPT INT8:")
    z = int8_matmul_dynamic_with_backward_ckpt(x, yqg, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = int8_matmul_dynamic_with_backward_ckpt(x, yqg, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_int8_dyn_ckpt_int8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ INT8 Dynamic CKPT INT8 test failed")
    sdnq_int8_dyn_ckpt_int8_tflops = 0


try:
    print("SDNQ INT8 Dynamic CKPT FP8:")
    z = int8_matmul_dynamic_with_backward_ckpt(x, yqgf, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = int8_matmul_dynamic_with_backward_ckpt(x, yqgf, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_int8_dyn_ckpt_fp8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ INT8 Dynamic CKPT FP8 test failed")
    sdnq_int8_dyn_ckpt_fp8_tflops = 0

try:
    print("SDNQ FP8 Dynamic CKPT Float:")
    z = fp8_matmul_dynamic_with_backward_ckpt(x, y, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_dynamic_with_backward_ckpt(x, y, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_dyn_ckpt_float_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 Dynamic CKPT Float test failed")
    sdnq_fp8_dyn_ckpt_float_tflops = 0


try:
    print("SDNQ FP8 Dynamic CKPT UINT8:")
    z = fp8_matmul_dynamic_with_backward_ckpt(x, yqgu, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_dynamic_with_backward_ckpt(x, yqgu, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_dyn_ckpt_uint8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 Dynamic CKPT UINT8 test failed")
    sdnq_fp8_dyn_ckpt_uint8_tflops = 0


try:
    print("SDNQ FP8 Dynamic CKPT INT8:")
    z = fp8_matmul_dynamic_with_backward_ckpt(x, yqg, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_dynamic_with_backward_ckpt(x, yqg, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_dyn_ckpt_int8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 Dynamic CKPT INT8 test failed")
    sdnq_fp8_dyn_ckpt_int8_tflops = 0


try:
    print("SDNQ FP8 Dynamic CKPT FP8:")
    z = fp8_matmul_dynamic_with_backward_ckpt(x, yqgf, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_dynamic_with_backward_ckpt(x, yqgf, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_dyn_ckpt_fp8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 Dynamic CKPT FP8 test failed")
    sdnq_fp8_dyn_ckpt_fp8_tflops = 0


try:
    print("SDNQ FP8 TW Dynamic CKPT Float:")
    z = fp8_matmul_tensorwise_dynamic_with_backward_ckpt(x, y, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_tensorwise_dynamic_with_backward_ckpt(x, y, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_tw_dyn_ckpt_float_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 TW Dynamic CKPT Float test failed")
    sdnq_fp8_tw_dyn_ckpt_float_tflops = 0


try:
    print("SDNQ FP8 TW Dynamic CKPT UINT8:")
    z = fp8_matmul_tensorwise_dynamic_with_backward_ckpt(x, yqgu, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_tensorwise_dynamic_with_backward_ckpt(x, yqgu, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_tw_dyn_ckpt_uint8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 TW Dynamic CKPT UINT8 test failed")
    sdnq_fp8_tw_dyn_ckpt_uint8_tflops = 0


try:
    print("SDNQ FP8 TW Dynamic CKPT INT8:")
    z = fp8_matmul_tensorwise_dynamic_with_backward_ckpt(x, yqg, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_tensorwise_dynamic_with_backward_ckpt(x, yqg, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_tw_dyn_ckpt_int8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 TW Dynamic CKPT INT8 test failed")
    sdnq_fp8_tw_dyn_ckpt_int8_tflops = 0


try:
    print("SDNQ FP8 TW Dynamic CKPT FP8:")
    z = fp8_matmul_tensorwise_dynamic_with_backward_ckpt(x, yqgf, b)
    loss = z.mean()
    loss.backward()
    sync_func()
    t0 = time.time()
    for i in tqdm(range(steps)):
        z = fp8_matmul_tensorwise_dynamic_with_backward_ckpt(x, yqgf, b)
        loss = z.mean()
        loss.backward()
        sync_func()
    t1 = time.time()
    sdnq_fp8_tw_dyn_ckpt_fp8_tflops = get_tflops(steps/(t1 - t0), m,n,k)
except Exception:
    print("SDNQ FP8 TW Dynamic CKPT FP8 test failed")
    sdnq_fp8_tw_dyn_ckpt_fp8_tflops = 0


print("\n")
print("===========================================")
print("GPU:", getattr(torch, torch.device(device).type).get_device_name(device))
print("Steps:", steps, "| MNK:", round((m*n*k)**(1/3)))
print("M:", m, "| N:", n, "| K:", k)
print("Float:", dtype)
print("===========================================")
print("PyTorch Float TFLOPS:", pytorch_float_tflops)
print("SDNQ Float TFLOPS:", sdnq_float_tflops)
print("SDNQ INT8 TFLOPS:", sdnq_int8_tflops)
print("SDNQ FP8 TFLOPS:", sdnq_fp8_tflops)
print("SDNQ FP8 TW TFLOPS:", sdnq_fp8_tw_tflops)
print("===========================================")
print("SDNQ Float UINT8 TFLOPS:", sdnq_float_uint8_tflops)
print("SDNQ Float INT8 TFLOPS:", sdnq_float_int8_tflops)
print("SDNQ Float FP8 TFLOPS:", sdnq_float_fp8_tflops)
print("===========================================")
print("SDNQ INT8 Dynamic Float TFLOPS:", sdnq_int8_dyn_float_tflops)
print("SDNQ INT8 Dynamic UINT8 TFLOPS:", sdnq_int8_dyn_uint8_tflops)
print("SDNQ INT8 Dynamic INT8 TFLOPS:", sdnq_int8_dyn_int8_tflops)
print("SDNQ INT8 Dynamic FP8 TFLOPS:", sdnq_int8_dyn_fp8_tflops)
print("===========================================")
print("SDNQ FP8 Dynamic Float TFLOPS:", sdnq_fp8_dyn_float_tflops)
print("SDNQ FP8 Dynamic UINT8 TFLOPS:", sdnq_fp8_dyn_uint8_tflops)
print("SDNQ FP8 Dynamic INT8 TFLOPS:", sdnq_fp8_dyn_int8_tflops)
print("SDNQ FP8 Dynamic FP8 TFLOPS:", sdnq_fp8_dyn_fp8_tflops)
print("===========================================")
print("SDNQ FP8 TW Dynamic Float TFLOPS:", sdnq_fp8_tw_dyn_float_tflops)
print("SDNQ FP8 TW Dynamic UINT8 TFLOPS:", sdnq_fp8_tw_dyn_uint8_tflops)
print("SDNQ FP8 TW Dynamic INT8 TFLOPS:", sdnq_fp8_tw_dyn_int8_tflops)
print("SDNQ FP8 TW Dynamic FP8 TFLOPS:", sdnq_fp8_tw_dyn_fp8_tflops)
print("===========================================")
print("SDNQ INT8 CKPT TFLOPS:", sdnq_int8_ckpt_tflops)
print("SDNQ FP8 CKPT TFLOPS:", sdnq_fp8_ckpt_tflops)
print("SDNQ FP8 TW CKPT TFLOPS:", sdnq_fp8_tw_ckpt_tflops)
print("===========================================")
print("SDNQ INT8 Dynamic CKPT Float TFLOPS:", sdnq_int8_dyn_ckpt_float_tflops)
print("SDNQ INT8 Dynamic CKPT UINT8 TFLOPS:", sdnq_int8_dyn_ckpt_uint8_tflops)
print("SDNQ INT8 Dynamic CKPT INT8 TFLOPS:", sdnq_int8_dyn_ckpt_int8_tflops)
print("SDNQ INT8 Dynamic CKPT FP8 TFLOPS:", sdnq_int8_dyn_ckpt_fp8_tflops)
print("===========================================")
print("SDNQ FP8 Dynamic CKPT Float TFLOPS:", sdnq_fp8_dyn_ckpt_float_tflops)
print("SDNQ FP8 Dynamic CKPT UINT8 TFLOPS:", sdnq_fp8_dyn_ckpt_uint8_tflops)
print("SDNQ FP8 Dynamic CKPT INT8 TFLOPS:", sdnq_fp8_dyn_ckpt_int8_tflops)
print("SDNQ FP8 Dynamic CKPT FP8 TFLOPS:", sdnq_fp8_dyn_ckpt_fp8_tflops)
print("===========================================")
print("SDNQ FP8 TW Dynamic CKPT Float TFLOPS:", sdnq_fp8_tw_dyn_ckpt_float_tflops)
print("SDNQ FP8 TW Dynamic CKPT UINT8 TFLOPS:", sdnq_fp8_tw_dyn_ckpt_uint8_tflops)
print("SDNQ FP8 TW Dynamic CKPT INT8 TFLOPS:", sdnq_fp8_tw_dyn_ckpt_int8_tflops)
print("SDNQ FP8 TW Dynamic CKPT FP8 TFLOPS:", sdnq_fp8_tw_dyn_ckpt_fp8_tflops)
print("===========================================")
print("\n")
