from typing import Mapping

from torch.utils.data import Dataset


class KdDatasetWrapper(Dataset):
    def __init__(self, **datasets: Dataset):
        """

        """
        self.datasets = datasets

        try:
            lengths = {key: len(dataset) for key, dataset in datasets.items()}
        except TypeError as e:
            raise TypeError("All datasets must implement __len__.") from e

        first_len = next(iter(lengths.values()))
        if any(L != first_len for L in lengths.values()):
            raise ValueError(f"All datasets must have the same length: {lengths}")

        self.datasets: Mapping[str, Dataset] = datasets
        self.length = first_len

    def __getitem__(self, idx: int) -> dict:
        return {key: ds[idx] for key, ds in self.datasets}

    def __len__(self):
        return self.length
