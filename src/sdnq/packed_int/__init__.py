import torch

from ..common import dtype_dict # noqa: TID252

from .pack import (
    pack_uint14,
    pack_uint12,
    pack_uint10,
    pack_uint7,
    pack_uint6,
    pack_uint5,
    pack_uint4,
    pack_uint3,
    pack_uint2,
    pack_uint1,
)

from .unpack import (
    unpack_uint14,
    unpack_uint12,
    unpack_uint10,
    unpack_uint7,
    unpack_uint6,
    unpack_uint5,
    unpack_uint4,
    unpack_uint3,
    unpack_uint2,
    unpack_uint1,
)


packed_int_function_dict = {
    "uint14": {"pack": pack_uint14, "unpack": unpack_uint14},
    "uint12": {"pack": pack_uint12, "unpack": unpack_uint12},
    "uint10": {"pack": pack_uint10, "unpack": unpack_uint10},
    "uint7": {"pack": pack_uint7, "unpack": unpack_uint7},
    "uint6": {"pack": pack_uint6, "unpack": unpack_uint6},
    "uint5": {"pack": pack_uint5, "unpack": unpack_uint5},
    "uint4": {"pack": pack_uint4, "unpack": unpack_uint4},
    "uint3": {"pack": pack_uint3, "unpack": unpack_uint3},
    "uint2": {"pack": pack_uint2, "unpack": unpack_uint2},
    "uint1": {"pack": pack_uint1, "unpack": unpack_uint1},
}


packed_int_function_dict["int14"] = packed_int_function_dict["uint14"]
packed_int_function_dict["int12"] = packed_int_function_dict["uint12"]
packed_int_function_dict["int10"] = packed_int_function_dict["uint10"]
packed_int_function_dict["int7"] = packed_int_function_dict["uint7"]
packed_int_function_dict["int6"] = packed_int_function_dict["uint6"]
packed_int_function_dict["int5"] = packed_int_function_dict["uint5"]
packed_int_function_dict["int4"] = packed_int_function_dict["uint4"]
packed_int_function_dict["int3"] = packed_int_function_dict["uint3"]
packed_int_function_dict["int2"] = packed_int_function_dict["uint2"]
packed_int_function_dict["bool"] = packed_int_function_dict["uint1"]


def pack_int(tensor: torch.Tensor, weights_dtype: str) -> torch.Tensor:
    if not dtype_dict[weights_dtype]["is_unsigned"]:
        tensor = tensor.sub(dtype_dict[weights_dtype]["min"])
    return packed_int_function_dict[weights_dtype]["pack"](tensor.to(dtype=dtype_dict[weights_dtype]["storage_dtype"]))


def unpack_int(packed_tensor: torch.Tensor, weights_dtype: str, shape: torch.Size, dtype: torch.dtype = None) -> torch.Tensor:
    packed_tensor = packed_int_function_dict[weights_dtype]["unpack"](packed_tensor, shape)
    if not dtype_dict[weights_dtype]["is_unsigned"]:
        packed_tensor = packed_tensor.to(dtype=dtype_dict[weights_dtype]["torch_dtype"] if dtype is None else dtype).add_(dtype_dict[weights_dtype]["min"])
    return packed_tensor
