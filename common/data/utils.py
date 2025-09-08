from typing import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader


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
