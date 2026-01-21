import torch


class SDNQLayer(torch.nn.Module):
    def __init__(self, original_layer, forward_func):
        torch.nn.Module.__init__(self)
        for key, value in original_layer.__dict__.items():
            if key not in {"forward", "forward_func", "original_class", "state_dict", "load_state_dict"}:
                setattr(self, key, value)
        self.original_class = original_layer.__class__
        self.forward_func = forward_func

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        if self.weight.__class__.__name__ == "SDNQTensor":
            if state_dict["weight"].zero_point is not None:
                state_dict.update({"weight.zero_point": state_dict["weight"].zero_point})
            if state_dict["weight"].svd_up is not None:
                state_dict.update({"weight.svd_up": state_dict["weight"].svd_up})
            if state_dict["weight"].svd_down is not None:
                state_dict.update({"weight.svd_down": state_dict["weight"].svd_down})
            state_dict.update({"weight": state_dict["weight"].weight, "weight.scale": state_dict["weight"].scale})
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        if self.weight.__class__.__name__ == "SDNQTensor":
            self.weight.weight = state_dict.pop("weight")
            self.weight.scale = state_dict.pop("weight.scale")
            self.weight.zero_point = state_dict.pop("weight.zero_point", self.weight.zero_point)
            self.weight.svd_up = state_dict.pop("weight.svd_up", self.weight.svd_up)
            self.weight.svd_down = state_dict.pop("weight.svd_down", self.weight.svd_down)
            state_dict["weight"] = self.weight
        return super().load_state_dict(state_dict, *args, **kwargs)


    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_func(self, *args, **kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(original_class={self.original_class.__name__} forward_func={self.forward_func} sdnq_dequantizer={repr(getattr(self, 'sdnq_dequantizer', None))})"


class SDNQLinear(SDNQLayer, torch.nn.Linear):
    original_class: torch.nn.Linear

class SDNQConv1d(SDNQLayer, torch.nn.Conv1d):
    original_class: torch.nn.Conv1d

class SDNQConv2d(SDNQLayer, torch.nn.Conv2d):
    original_class: torch.nn.Conv2d

class SDNQConv3d(SDNQLayer, torch.nn.Conv3d):
    original_class: torch.nn.Conv3d

class SDNQConvTranspose1d(SDNQLayer, torch.nn.ConvTranspose1d):
    original_class: torch.nn.ConvTranspose1d

class SDNQConvTranspose2d(SDNQLayer, torch.nn.ConvTranspose2d):
    original_class: torch.nn.ConvTranspose2d

class SDNQConvTranspose3d(SDNQLayer, torch.nn.ConvTranspose3d):
    original_class: torch.nn.ConvTranspose3d


torch.serialization.add_safe_globals([SDNQLayer])
torch.serialization.add_safe_globals([SDNQLinear])
torch.serialization.add_safe_globals([SDNQConv1d])
torch.serialization.add_safe_globals([SDNQConv2d])
torch.serialization.add_safe_globals([SDNQConv3d])
torch.serialization.add_safe_globals([SDNQConvTranspose1d])
torch.serialization.add_safe_globals([SDNQConvTranspose2d])
torch.serialization.add_safe_globals([SDNQConvTranspose3d])


def get_sdnq_wrapper_class(original_layer, forward_func):
    match original_layer.__class__.__name__:
        case "Linear":
            return SDNQLinear(original_layer, forward_func)
        case "Conv1d":
            return SDNQConv1d(original_layer, forward_func)
        case "Conv2d":
            return SDNQConv2d(original_layer, forward_func)
        case "Conv3d":
            return SDNQConv3d(original_layer, forward_func)
        case "ConvTranspose1d":
            return SDNQConvTranspose1d(original_layer, forward_func)
        case "ConvTranspose2d":
            return SDNQConvTranspose2d(original_layer, forward_func)
        case "ConvTranspose3d":
            return SDNQConvTranspose3d(original_layer, forward_func)
        case _:
            return SDNQLayer(original_layer, forward_func)
