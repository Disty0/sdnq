# pylint: disable=redefined-builtin,no-member,protected-access

import os
import logging
import torch

logger = logging.getLogger("sdnq")

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
            self.backend = "ipex"
        elif self.backend == "cuda" and torch.version.hip is not None:
            self.backend = "rocm"

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
                try:
                    getattr(torch, torch.device(self.device).type).synchronize()
                    getattr(torch, torch.device(self.device).type).empty_cache()
                except Exception:
                    pass

    def has_triton(self) -> bool:
        try:
            if torch._dynamo.config.disable:
                triton_is_available = False
            else:
                from torch.utils._triton import has_triton as torch_has_triton
                triton_is_available = torch_has_triton()
        except Exception:
            triton_is_available = False
        if triton_is_available:
            backup_suppress_errors = torch._dynamo.config.suppress_errors
            torch._dynamo.config.suppress_errors = False
            try:
                def test_triton_func(a,b,c):
                    return a * b + c
                test_triton_func = torch.compile(test_triton_func, fullgraph=True)
                test_triton_func(torch.randn(32, device=self.device), torch.randn(32, device=self.device), torch.randn(32, device=self.device))
                triton_is_available = True
            except Exception as e:
                triton_is_available = False
                logger.warning(f"SDNQ: Triton test failed! Falling back to PyTorch Eager mode. Error message: {e}")
            torch._dynamo.config.suppress_errors = backup_suppress_errors
        else:
            logger.warning("SDNQ: Triton is not available. Falling back to PyTorch Eager mode.")
        return triton_is_available


class SharedOpts():
    def __init__(self, devices):
        self.diffusers_offload_mode = os.environ.get("SDNQ_OFFLOAD_MODE", "none").lower()
        if os.environ.get("SDNQ_USE_TORCH_COMPILE", None) is None:
            self.sdnq_dequantize_compile = devices.has_triton()
        else:
            self.sdnq_dequantize_compile = bool(os.environ.get("SDNQ_USE_TORCH_COMPILE", "1").lower() not in {"0", "false", "no"})


class Shared():
    def __init__(self, devices, logger):
        self.log = logger
        self.opts = SharedOpts(devices=devices)


devices = Devices()
shared = Shared(devices=devices, logger=logger)
