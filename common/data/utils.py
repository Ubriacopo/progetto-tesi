from typing import OrderedDict

import torch
from torch.utils.data import Dataset, DataLoader


class BoundedMap(OrderedDict):
    def __init__(self, max_length: int, lru: bool = False):
        super().__init__()
        self.max_length, self.lru = max_length, lru

    def __getitem__(self, key):
        v = super().__getitem__(key)
        if self.lru:
            self.move_to_end(key)  # mark as most-recent
        return v

    def __setitem__(self, key, value):
        if key in self and self.lru:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.max_length:
            # Evict the oldest entry
            self.popitem(last=False)


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
