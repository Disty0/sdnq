import torch


class SDNQLayer(torch.nn.Module):
    def __init__(self, original_layer, forward_func):
        super().__init__()
        for key, value in original_layer.__dict__.items():
            setattr(self, key, value)
        self.parent_class = original_layer.__class__
        self.forward_func = forward_func

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.forward_func(self, *args, **kwargs)

torch.serialization.add_safe_globals([SDNQLayer])
