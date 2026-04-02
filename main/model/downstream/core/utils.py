import torch
from tensordict import TensorDict


def dequantize(x: dict | TensorDict, dequantize_keys: list[str], dtype=torch.float16):
    return_dict: dict = {}
    for key, td in x.items():
        if key in dequantize_keys and "data" in td:
            data = td["data"].to(dtype=dtype)
            data.mul_(td["scales"])
            td = {"data": data, "mask": td["mask"]}
        return_dict[key] = td
    return TensorDict.from_dict(return_dict, auto_batch_size=True)
