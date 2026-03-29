from typing import Callable

from tensordict import TensorDict, tensordict

from main.core_data.ds.td_dataset import TdSegmentedExperimentDataset
from main.model.downstream.base import BaseDatamodule


class FacedDataModule(BaseDatamodule):
    def __init__(self, faced_path: str, seed: int, batch_size: int):
        super().__init__(seed=seed, batch_size=batch_size)
        self.add_dataset(faced_path, valid_fraction=0.1, test_fraction=0.15)

    def merge_with_dataset(self, dataset, ds_path: str, use_ids: list, weight=1):
        return TdSegmentedExperimentDataset(ds_path, ds_path + "/spec.csv", use_ids)

    @staticmethod
    def train_collate_fn(batch: list[TensorDict]) -> Callable[[list[TensorDict]], TensorDict]:
        batch = [b.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )[:10] for b in batch]
        return tensordict.pad_sequence(batch, 0, return_mask="pad_mask")

    @staticmethod
    def test_collate_fn(batch: list[TensorDict]) -> Callable:
        batch = [b.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )[:10] for b in batch]
        return tensordict.pad_sequence(batch, 0, return_mask="pad_mask")

    @staticmethod
    def valid_collate_fn(batch: list[TensorDict]) -> Callable:
        batch = [b.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )[:10] for b in batch]
        return tensordict.pad_sequence(batch, 0, return_mask="pad_mask")
