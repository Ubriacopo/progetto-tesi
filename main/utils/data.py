from typing import TypedDict, Optional

import numpy as np
import torch


def pad_main_axis(arr: np.ndarray, increment: int, value=0) -> np.ndarray:
    paddings = [(0, 0)] * arr.ndim
    # Work on main axis
    paddings[0] = (0, increment)
    return np.pad(arr, paddings, constant_values=value)


class MaskedValue(TypedDict):
    data: torch.Tensor
    mask: Optional[torch.Tensor]

class KdMaskedValue(MaskedValue, total=False):
    kd: MaskedValue | torch.Tensor
    t_mod: MaskedValue | torch.Tensor
