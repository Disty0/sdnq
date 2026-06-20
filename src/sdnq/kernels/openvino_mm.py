import os
import torch
import openvino as ov
from openvino import opset16 as ov_ops
from openvino.properties import hint as ov_hints

core = ov.Core()

OV_DEVICE: str = os.environ.get("SDNQ_OPENVINO_DEVICE", "CPU")
OV_COMPILED_CACHE: dict[tuple[str, tuple[int,int] | None, tuple[int,int] | None], tuple[ov.InferRequest, str]] = {}

if OV_DEVICE == "NPU":
    OV_DEVICE = "HETERO:NPU,CPU"

if OV_DEVICE.startswith("HETERO:"):
    for ov_device in OV_DEVICE.removeprefix("HETERO:").split(","):
        core.set_property(ov_device, {ov_hints.execution_mode: ov_hints.ExecutionMode.ACCURACY})
else:
    core.set_property(OV_DEVICE, {ov_hints.execution_mode: ov_hints.ExecutionMode.ACCURACY})


def ov_int_mm(A: torch.CharTensor, B: torch.CharTensor, infer_request: ov.InferRequest, out_name: str) -> torch.FloatTensor:
    C = torch.empty((A.shape[0], B.shape[-1]), device="cpu", dtype=torch.float32)
    infer_request.set_tensor("A", ov.Tensor(A.detach().contiguous().to("cpu").numpy(), shared_memory=True))
    infer_request.set_tensor("B", ov.Tensor(B.detach().contiguous().to("cpu").numpy(), shared_memory=True))
    infer_request.set_tensor(out_name, ov.Tensor(C.numpy(), shared_memory=True))
    infer_request.infer()
    C = C.to(A.device)
    return C


@torch.library.custom_op("sdnq::openvino_int_mm", mutates_args=())
def openvino_int_mm(Tensor_A: torch.Tensor, Tensor_B: torch.Tensor) -> torch.Tensor:
    if "GPU" not in OV_DEVICE:
        cache_key = (OV_DEVICE, Tensor_A.shape, Tensor_B.shape)
    else:
        cache_key = (OV_DEVICE, None, None)
    infer_request, out_name = OV_COMPILED_CACHE.get(cache_key, (None, None))
    if infer_request is not None:
        return ov_int_mm(Tensor_A, Tensor_B, infer_request, out_name)

    if "GPU" not in OV_DEVICE:
        shape_a = ov.Shape(Tensor_A.shape)
        shape_b = ov.Shape(Tensor_B.shape)
    else:
        shape_a = ov.PartialShape([-1,-1])
        shape_b = ov.PartialShape([-1,-1])
    input_a = ov_ops.parameter(shape_a, ov.Type.i8, name="A")
    input_b = ov_ops.parameter(shape_b, ov.Type.i8, name="B")

    low  = ov_ops.constant(-128.0, dtype=ov.Type.f32)
    high = ov_ops.constant(127.0,  dtype=ov.Type.f32)
    a = ov_ops.fake_quantize(ov_ops.convert(input_a, ov.Type.f32), low, high, low, high, 256)
    b = ov_ops.fake_quantize(ov_ops.convert(input_b, ov.Type.f32), low, high, low, high, 256)

    # NPU uses FP16 x INT8 -> FP16 instead of INT8 x INT8 -> INT32 and FP16 output overflows
    if "NPU" in OV_DEVICE:
        npu_mul = ov_ops.constant(Tensor_B.shape[-2] ** 0.5, dtype=ov.Type.f32)
        npu_mul_out = ov_ops.constant(Tensor_B.shape[-2], dtype=ov.Type.f32, name="npu_out_mul_constant")
        a = ov_ops.divide(a, npu_mul)
        b = ov_ops.divide(b, npu_mul)
        out = ov_ops.matmul(a, b, False, False)
        out = ov_ops.multiply(out, npu_mul_out, name="npu_out_mul")
    else:
        out = ov_ops.matmul(a, b, False, False)

    ov_model = ov.Model([out], [input_a, input_b], "ov_int8_mm")
    if "NPU" in OV_DEVICE: # NPU can't use FP32 for regular multiplications
        for node in ov_model.get_ops():
            if node.get_friendly_name() in {"npu_out_mul", "npu_out_mul_constant"}:
                node.get_rt_info()["affinity"] = "CPU"
            else:
                node.get_rt_info()["affinity"] = "NPU"
    ov_model = core.compile_model(ov_model, OV_DEVICE)
    infer_request = ov_model.create_infer_request()
    out_name = ov_model.outputs[0]

    OV_COMPILED_CACHE[cache_key] = (infer_request, out_name)
    return ov_int_mm(Tensor_A, Tensor_B, infer_request, out_name)

@openvino_int_mm.register_fake
def openvino_int_mm_fake(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.mm(A.to(dtype=torch.float32), B.to(dtype=torch.float32))
