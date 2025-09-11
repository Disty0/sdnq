import torch
from safetensors.torch import save_file
from sdnq.training import SDNQTensor


def main(model_path, out_path, dtype=None):
    print("\nLoading the SDNQ model...\n")
    state_dict = torch.load(model_path, map_location="cpu")
    for key, value in state_dict.items():
        if isinstance(value, SDNQTensor):
            print("Dequantizing:", key)
            value.sr = False
            value.return_dtype = dtype if dtype is not None else getattr(value, "return_dtype", value.scale.dtype)
            state_dict[key] = value.dequantize()
    print("\nSaving the converted the model...")
    if out_path.endswith(".safetensors"):
        save_file(state_dict, out_path)
    else:
        torch.save(state_dict, out_path)
    print("Successfully converted the model!\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create a bucket list with a given dataset path')
    parser.add_argument('model_path', type=str)
    parser.add_argument('out_path', type=str)
    parser.add_argument('--dtype', default="none", type=str)

    args = parser.parse_args()
    dtype = getattr(torch, args.dtype) if args.dtype not in {None, "none"} else None
    main(args.model_path, args.out_path, dtype=dtype)
