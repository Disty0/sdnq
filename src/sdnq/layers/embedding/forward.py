# pylint: disable=relative-beyond-top-level,redefined-builtin,protected-access

import torch


def quantized_embedding_forward(self, input: torch.FloatTensor) -> torch.FloatTensor:
    return torch.nn.functional.embedding(
        input,
        self.sdnq_dequantizer(self.weight, self.scale, self.zero_point, self.svd_up, self.svd_down),
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse,
    )
