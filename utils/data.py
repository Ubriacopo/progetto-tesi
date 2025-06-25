from pathlib import Path

import numpy as np
from scipy.io import loadmat


def extract_trial_data(destination_path: str, source_path: str):
    """
    The data is converted to a numpy friendly type to help us work better (we have some advantages).
    Consideration:
        Load all EEG (40 × 100MB = ~4GB) into RAM at startup — easy with 128GB. (Our server)

    :param destination_path:
    :param source_path:
    """
    mat = loadmat(source_path)  # Source file

    data = {k: v for k, v in mat.items() if not k.startswith("__")}
    for key in data:
        # Remove the heading dimension
        data[key] = data[key].squeeze()

    file_name = Path(source_path).stem

    output_path = Path(f"{destination_path}/{file_name}.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)
