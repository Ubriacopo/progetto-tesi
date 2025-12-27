from __future__ import annotations

import dataclasses
from datetime import datetime
import gzip
import shutil
import time
from functools import wraps
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BatchFeature

from base_config import BaseConfig
from main.utils.logging import make_logger

utils_logger = make_logger("data.utils")


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
            utils_logger.warn("String data won't be persisted. Only tensors")
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


def debug_exceptional_catch(func):
    exceptional_catch_logger = make_logger(debug_exceptional_catch.__name__)

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            exceptional_catch_logger.error(f"{func.__name__}: raised exception: \n")
            exceptional_catch_logger.error(e)
            raise e

    return wrapper


def call_log(before: bool = True, after: bool = False, suppress: bool = BaseConfig.SUPPRESS_ENTER_LEAVE_LOG):
    def decorator(fn):
        # Disable the function entirely
        if suppress:
            return fn

        @wraps(fn)
        def wrapper(*args, **kwargs):
            name = fn.__name__
            classname = "[function]"
            if args and hasattr(args[0], "__class__"):
                classname = f"[{args[0].__class__.__name__}]"

            logger = make_logger(name)
            before and logger.info(f"Entering {classname}.{name}")
            result = fn(*args, **kwargs)
            after and logger.info(f"Exiting {classname}.{name}")
            return result

        return wrapper

    return decorator


def timed(label: str = None, longer_than: float = 0.5, suppress: bool = BaseConfig.SUPPRESS_TIMED):
    def decorator(fn):
        # Disable the function entirely
        if suppress:
            return fn

        logger = make_logger("timed")

        @wraps(fn)
        def wrapper(*args, **kwargs):
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
                logger.debug(
                    f"{datetime.today().strftime('%H:%M:%S')}:{tag} took {end - start:.3f} seconds ({start:.2f} - {end:.2f})"
                )
            return result

        return wrapper

    return decorator
