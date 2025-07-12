from pathlib import Path

import numpy as np
from scipy.io import loadmat


def pad_main_axis(arr: np.ndarray, increment: int, value=0) -> np.ndarray:
    paddings = [(0, 0)] * arr.ndim
    # Work on main axis
    paddings[0] = (0, increment)
    return np.pad(arr, paddings, constant_values=value)
