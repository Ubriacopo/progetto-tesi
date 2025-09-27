from __future__ import annotations

import dataclasses
from datetime import datetime
import gzip
import shutil
import time
from contextlib import suppress
from functools import wraps
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BatchFeature

from base_config import BaseConfig


def dataset_information(dataset: Dataset, image_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the mean and variance of the data.
    I followed the example shown here: https://kozodoi.me/blog/20210308/compute-image-stats
    :param image_size: The size of the images of the data in input
    :param dataset: Dataset to measure mean and standard deviation of
    :return: the mean and standard deviation of the data
    """
    sums = torch.tensor([0.0, 0.0, 0.0])
    square_sums = torch.tensor([0.0, 0.0, 0.0])

    dataloader = DataLoader(dataset, batch_size=None, num_workers=0, shuffle=False)
    size = len(dataloader) * image_size[0] * image_size[1]

    for image, _ in dataloader:
        sums += image.sum(axis=(1, 2))
        square_sums += (image ** 2).sum(axis=(1, 2))

    mean = sums / size  # Mean
    variance = square_sums / size - mean ** 2

    return mean, variance


def build_tensor_dict(samples: list[dict | torch.Tensor] | tuple):
    """
        torch.save({
            'eeg': torch.stack([sample['eeg'] for sample in a]),  # (17, 14, 8, 200)
            'vid': torch.stack([sample['vid'] for sample in a]),   # (17, 400)
            'aud': {
                "in":torch.stack([sample['aud']["input_features"] for sample in a]),
                "attn": torch.stack([sample['aud']["attention_mask"] for sample in a])
            }
        }, 'batched_file.pt')
    This is kinda efficient and I can keep structured data to feed the model.

    :param samples: Objects to store. Fields that are str are ignored. Unrecognizable types beyond tensors, nums, list and dicts raise exceptions.
    :return: The build tensor dictionary where samples are stacked together.
    """
    try:
        first = samples[0]
        if isinstance(first, torch.Tensor):
            return torch.stack(samples)

        if isinstance(first, dict) or isinstance(first, BatchFeature):
            return {k: build_tensor_dict([s[k] for s in samples]) for k in first.keys()}

        if isinstance(first, (list, tuple)):
            return type(first)(build_tensor_dict(items) for items in zip(*samples))

        if isinstance(first, str):
            print("String data won't be persisted. Only tensors")
            return torch.empty(0)

        else:
            raise TypeError(f"Unsupported type: {type(first)}")
    except Exception as e:
        raise e


def sanitize_for_ast(obj):
    # primitives already fine
    if isinstance(obj, _AST_OK):
        return obj

    # Numpy scalars -> Python scalars
    if isinstance(obj, np.generic):
        return obj.item()
    # Numpy arrays -> Nested lists (0-d -> scalar)
    if isinstance(obj, np.ndarray):
        return obj.item() if obj.ndim == 0 else obj.tolist()
    # Torch tensors -> Nested lists (0-d -> scalar)
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.tolist()

    # Dataclass -> To Dict First
    if dataclasses.is_dataclass(obj):
        return sanitize_for_ast(dataclasses.asdict(obj))

    # Mappings
    if isinstance(obj, Mapping):
        return {(k if isinstance(k, _AST_OK) else str(k)): sanitize_for_ast(v) for k, v in obj.items()}

    # Sequences (but not str/bytes which were caught above)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        typ = tuple if isinstance(obj, tuple) else list
        return typ(sanitize_for_ast(x) for x in obj)

    # Sets
    if isinstance(obj, set):
        return {sanitize_for_ast(x) for x in obj}

    # Default is str representation
    return str(obj)


_AST_OK = (str, bytes, bool, int, float, type(None))


def compress_pt_in_folder(path_to_dir: str, verbose: bool = True):
    if not Path(path_to_dir).exists():
        raise FileNotFoundError(f"Folder {path_to_dir} does not exist")
    if not Path(path_to_dir).is_dir():
        raise NotADirectoryError(f"Folder {path_to_dir} is not a directory")

    path = Path(path_to_dir)
    pt_files = [file for file in path.iterdir() if file.glob("*.pt")]
    for pt_file in pt_files:
        print("Compressing file {}".format(pt_file)) if verbose else None
        with open(pt_file, "rb") as f_in, gzip.open(Path(str(pt_file) + ".gz"), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("Compressed file {}".format(pt_file)) if verbose else None


def decompress_pt(path_to_pt: str, map_location: str = "cpu"):
    gz_file_path = Path(path_to_pt).with_suffix(".gz")
    if not gz_file_path.exists():
        raise FileNotFoundError(f"File {str(gz_file_path)} does not exist")

    with gzip.open(gz_file_path, "rb") as f_in:
        return torch.load(f_in, map_location=map_location)


def timed(label: str = None, longer_than: float = 0.5, suppress_timed: bool = BaseConfig.SUPPRESS_TIMED):
    def decorator(fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Disable the function entirely
            if suppress_timed:
                return fn(*args, **kwargs)

            start = time.perf_counter()
            result = fn(*args, **kwargs)
            end = time.perf_counter()

            # If called on a class instance, use its class name
            if args and hasattr(args[0], "__class__"):
                cls_name = args[0].__class__.__name__
            else:
                cls_name = fn.__name__

            tag = label or f"{cls_name}.{fn.__name__}"
            # Maybe really short times are ignorable
            if longer_than < end - start:
                print(f"{datetime.today().strftime('%Y-%m-%d')}:{tag} took {end - start:.3f} seconds ({start} - {end})")
            return result

        return wrapper

    return decorator
