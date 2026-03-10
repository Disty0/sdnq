import torch


def unpack_uint14(packed_tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = torch.cat(
        (
            torch.bitwise_and(packed_tensor[:, :7], 16383),
            torch.bitwise_or(
                torch.bitwise_or(
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 0], 2), 12288),
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 1], 4), 3072),
                    ),
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 2], 6), 768),
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 3], 8), 192),
                    ),
                ),
                torch.bitwise_or(
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 4], 10), 48),
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 5], 12), 12),
                    ),
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 6], 14), 3),
                ),
            ).unsqueeze(-1)
        ),
        dim=-1
    ).view(shape)
    return result


def unpack_uint12(packed_tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = torch.cat(
        (
            torch.bitwise_and(packed_tensor[:, :3], 4095),
            torch.bitwise_or(
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 0], 4), 3840),
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 1], 8), 240),
                ),
                torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 2], 12), 15)
            ).unsqueeze(-1)
        ),
        dim=-1
    ).view(shape)
    return result


def unpack_uint10(packed_tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result_bitwise_right_shift = torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, :3], 10), 63)
    result = torch.cat(
        (
            torch.bitwise_and(packed_tensor[:, :5], 1023),
            torch.bitwise_or(
                result_bitwise_right_shift[:, :2],
                torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 3:5], 4), 960),
            ),
            torch.bitwise_or(
                result_bitwise_right_shift[:, 2],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 3], 6), 768),
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 4], 8), 192),
                ),
            ).unsqueeze(-1),
        ),
        dim=-1
    ).view(shape)
    return result


def unpack_uint7(packed_tensor: torch.ByteTensor, shape: torch.Size) -> torch.ByteTensor:
    result = torch.cat(
        (
            torch.bitwise_and(packed_tensor[:, :7], 127),
            torch.bitwise_or(
                torch.bitwise_or(
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 0], 1), 64),
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 1], 2), 32),
                    ),
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 2], 3), 16),
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 3], 4), 8),
                    ),
                ),
                torch.bitwise_or(
                    torch.bitwise_or(
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 4], 5), 4),
                        torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 5], 6), 2),
                    ),
                    torch.bitwise_right_shift(packed_tensor[:, 6], 7),
                ),
            ).unsqueeze(-1)
        ),
        dim=-1
    ).view(shape)
    return result


def unpack_uint6(packed_tensor: torch.ByteTensor, shape: torch.Size) -> torch.ByteTensor:
    result = torch.cat(
        (
            torch.bitwise_and(packed_tensor[:, :3], 63),
            torch.bitwise_or(
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 0], 2), 48),
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 1], 4), 12),
                ),
                torch.bitwise_right_shift(packed_tensor[:, 2], 6)
            ).unsqueeze(-1)
        ),
        dim=-1
    ).view(shape)
    return result


def unpack_uint5(packed_tensor: torch.ByteTensor, shape: torch.Size) -> torch.ByteTensor:
    result_bitwise_right_shift = torch.bitwise_right_shift(packed_tensor[:, :3], 5)
    result = torch.cat(
        (
            torch.bitwise_and(packed_tensor[:, :5], 31),
            torch.bitwise_or(
                result_bitwise_right_shift[:, :2],
                torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 3:5], 2), 24),
            ),
            torch.bitwise_or(
                result_bitwise_right_shift[:, 2],
                torch.bitwise_or(
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 3], 3), 16),
                    torch.bitwise_and(torch.bitwise_right_shift(packed_tensor[:, 4], 4), 8),
                ),
            ).unsqueeze(-1),
        ),
        dim=-1
    ).view(shape)
    return result


def unpack_uint4(packed_tensor: torch.ByteTensor, shape: torch.Size) -> torch.ByteTensor:
    result = torch.stack((torch.bitwise_and(packed_tensor, 15), torch.bitwise_right_shift(packed_tensor, 4)), dim=-1).view(shape)
    return result


def unpack_uint3(packed_tensor: torch.ByteTensor, shape: torch.Size) -> torch.ByteTensor:
    result = torch.bitwise_and(
        torch.cat(
            (
                packed_tensor[:, :3],
                torch.bitwise_right_shift(packed_tensor[:, :3], 3),
                torch.bitwise_or(
                    torch.bitwise_right_shift(packed_tensor[:, :2], 6),
                    torch.bitwise_and(
                        torch.stack(
                            (
                                torch.bitwise_right_shift(packed_tensor[:, 2], 4),
                                torch.bitwise_right_shift(packed_tensor[:, 2], 5),
                            ),
                            dim=-1
                        ),
                        4
                    ),
                ),
            ),
            dim=-1
        ),
        7
    ).view(shape)
    return result


def unpack_uint2(packed_tensor: torch.ByteTensor, shape: torch.Size) -> torch.ByteTensor:
    result = torch.bitwise_and(
        torch.stack(
            (
                packed_tensor,
                torch.bitwise_right_shift(packed_tensor, 2),
                torch.bitwise_right_shift(packed_tensor, 4),
                torch.bitwise_right_shift(packed_tensor, 6),
            ),
            dim=-1
        ),
        3
    ).view(shape)
    return result


def unpack_uint1(packed_tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    result = torch.bitwise_and(
        torch.stack(
            (
                packed_tensor,
                torch.bitwise_right_shift(packed_tensor, 1),
                torch.bitwise_right_shift(packed_tensor, 2),
                torch.bitwise_right_shift(packed_tensor, 3),
                torch.bitwise_right_shift(packed_tensor, 4),
                torch.bitwise_right_shift(packed_tensor, 5),
                torch.bitwise_right_shift(packed_tensor, 6),
                torch.bitwise_right_shift(packed_tensor, 7),
            ),
            dim=-1
        ),
        1
    ).view(shape)
    return result
