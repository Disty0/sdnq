# pylint: disable=redefined-builtin,no-member,protected-access

import os
import torch

# wrapper for modules.devices and modules.shared from SD.Next
class Devices():
    def __init__(self):
        self.cpu = torch.device("cpu")
        self.device = torch.device(
            os.environ.get("SDNQ_DEVICE",
                "xpu" if hasattr(torch,"xpu") and torch.xpu.is_available()
                else "mps" if hasattr(torch,"mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available()
                else "cuda" if torch.cuda.is_available()
                else "cpu"
            ).lower()
        )
        self.backend = self.device.type
        self.dtype = getattr(torch, os.environ.get("SDNQ_DTYPE", "bfloat16" if self.backend != "cpu" else "float32"))
        self.inference_context = torch.no_grad
        if self.backend == "xpu":
            self.backend == "ipex"

    def normalize_device(self, dev):
        if torch.device(dev).type in {"cpu", "mps", "meta"}:
            return torch.device(dev)
        if torch.device(dev).index is None:
            return torch.device(str(dev), index=0)
        return torch.device(dev)

    def same_device(self, d1, d2):
        return self.normalize_device(d1) == self.normalize_device(d2)

    def torch_gc(self, force:bool=False, fast:bool=False, reason:str=None):
        if force:
            import gc
            gc.collect()
            if self.backend != "cpu":
                getattr(torch, self.backend).synchronize()
                getattr(torch, self.backend).empty_cache()

    def has_triton(self) -> bool:
        try:
            from torch.utils._triton import has_triton as torch_has_triton
            return torch_has_triton()
        except Exception:
            return False


class SharedOpts():
    def __init__(self, backend):
        self.diffusers_offload_mode = os.environ.get("SDNQ_OFFLOAD_MODE", "none").lower()
        self.sdnq_dequantize_compile = devices.has_triton() and os.environ.get("SDNQ_USE_TORCH_COMPILE", "1").lower() not in {"0", "false", "no"}


class Shared():
    def __init__(self, backend):
        self.opts = SharedOpts(backend=backend)


devices = Devices()
shared = Shared(backend=devices.backend)
