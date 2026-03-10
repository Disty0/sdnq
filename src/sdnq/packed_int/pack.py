import torch


def pack_uint14(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, :7],
        torch.bitwise_and(
            torch.stack(
                (
                    torch.bitwise_left_shift(packed_tensor[:, 7], 2),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 4),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 6),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 8),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 10),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 12),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 14),
                ),
                dim=-1
            ),
            49152
        ),
    )
    return packed_tensor


def pack_uint12(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 4)
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, :3],
        torch.bitwise_and(
            torch.stack(
                (
                    torch.bitwise_left_shift(packed_tensor[:, 3], 4),
                    torch.bitwise_left_shift(packed_tensor[:, 3], 8),
                    torch.bitwise_left_shift(packed_tensor[:, 3], 12),
                ),
                dim=-1
            ),
            61440
        )
    )
    return packed_tensor


def pack_uint10(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.cat(
        (
            torch.bitwise_or(packed_tensor[:, :3], torch.bitwise_left_shift(packed_tensor[:, 5:8], 10)),
            torch.bitwise_or(
                packed_tensor[:, 3],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 5], 4), 15360),
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 7], 6), 49152),
                ),
            ).unsqueeze(-1),
            torch.bitwise_or(
                packed_tensor[:, 4],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 6], 4), 15360),
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 7], 8), 49152),
                ),
            ).unsqueeze(-1),
        ),
        dim=-1
    )
    return packed_tensor


def pack_uint7(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, :7],
        torch.bitwise_and(
            torch.stack(
                (
                    torch.bitwise_left_shift(packed_tensor[:, 7], 1),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 2),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 3),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 4),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 5),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 6),
                    torch.bitwise_left_shift(packed_tensor[:, 7], 7),
                ),
                dim=-1
            ),
            128
        ),
    )
    return packed_tensor


def pack_uint6(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 4)
    packed_tensor = torch.bitwise_or(
        packed_tensor[:, :3],
        torch.bitwise_and(
            torch.stack(
                (
                    torch.bitwise_left_shift(packed_tensor[:, 3], 2),
                    torch.bitwise_left_shift(packed_tensor[:, 3], 4),
                    torch.bitwise_left_shift(packed_tensor[:, 3], 6),
                ),
                dim=-1
            ),
            192
        )
    )
    return packed_tensor


def pack_uint5(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.cat(
        (
            torch.bitwise_or(packed_tensor[:, :3], torch.bitwise_left_shift(packed_tensor[:, 5:8], 5)),
            torch.bitwise_or(
                packed_tensor[:, 3],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 5], 2), 96),
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 7], 3), 128),
                ),
            ).unsqueeze(-1),
            torch.bitwise_or(
                packed_tensor[:, 4],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 6], 2), 96),
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 7], 4), 128),
                ),
            ).unsqueeze(-1),
        ),
        dim=-1
    )
    return packed_tensor


def pack_uint4(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 2)
    packed_tensor = torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 4))
    return packed_tensor


def pack_uint3(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(packed_tensor[:, :3], torch.bitwise_left_shift(packed_tensor[:, 3:6], 3)),
        torch.cat(
            (
                torch.bitwise_left_shift(packed_tensor[:, 6:8], 6),
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 6], 4), 64),
                    torch.bitwise_and(torch.bitwise_left_shift(packed_tensor[:, 7], 5), 128),
                ).unsqueeze(-1),
            ),
            dim=-1
        )
    )
    return packed_tensor


def pack_uint2(tensor: torch.ByteTensor) -> torch.ByteTensor:
    packed_tensor = tensor.contiguous().view(-1, 4)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 2)),
        torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 2], 4), torch.bitwise_left_shift(packed_tensor[:, 3], 6)),
    )
    return packed_tensor


def pack_uint1(tensor: torch.Tensor) -> torch.Tensor:
    packed_tensor = tensor.contiguous().view(-1, 8)
    packed_tensor = torch.bitwise_or(
        torch.bitwise_or(
            torch.bitwise_or(packed_tensor[:, 0], torch.bitwise_left_shift(packed_tensor[:, 1], 1)),
            torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 2], 2), torch.bitwise_left_shift(packed_tensor[:, 3], 3))
        ),
        torch.bitwise_or(
            torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 4], 4), torch.bitwise_left_shift(packed_tensor[:, 5], 5)),
            torch.bitwise_or(torch.bitwise_left_shift(packed_tensor[:, 6], 6), torch.bitwise_left_shift(packed_tensor[:, 7], 7))
        ),
    )
    return packed_tensor
