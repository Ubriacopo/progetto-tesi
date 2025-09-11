from typing import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BatchFeature


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
