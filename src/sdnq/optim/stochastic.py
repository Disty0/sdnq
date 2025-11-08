import torch
from torch import Tensor

# Modified from: https://github.com/Nerogar/OneTrainer/blob/master/modules/util/bf16_stochastic_rounding.py


def copy_stochastic_(target: Tensor, source: Tensor):
    """
    copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16 or float16
        source: the target tensor with dtype=float32
    """
    assert target.dtype in {torch.float16, torch.bfloat16}
    if source.dtype != torch.float32:
        source = source.to(dtype=torch.float32)

    # mantissa_mask = 1 << (fp32_mantissa_bits - target_mantissa_bits)
    mantissa_difference = 8192 if target.dtype == torch.float16 else 65536

    # create a random integer for the missin part of the mantissa
    result = torch.randint_like(source, dtype=torch.int32, low=0, high=mantissa_difference)

    # add the random number to the lower part of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower part of the mantissa
    result.bitwise_and_(-mantissa_difference)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result
