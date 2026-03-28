from pathlib import Path
from typing import Optional, Callable, Any

import lightning
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from main.core_data.ds.td_dataset import TdSegmentedExperimentDataset
from main.utils.logging import make_logger

class LinearProbeDataModule(lightning.LightningDataModule):
    def __init__(self, seed: int, batch_size: int):
        super().__init__()
        self.logger = make_logger(self.__class__.name)

        self.seed: int = seed
        self.batch_size: int = batch_size

        # Where dataset references are stored
        self.train_dataset: Optional[Dataset] | list[Dataset] = []
        self.train_collate_fn: Optional[Callable[[Any], Any]] = None
        self.train_dataset_weight: list[int] = []

        self.valid_dataset: Optional[Dataset] | list[Dataset] = []
        self.valid_collate_fn: Optional[Callable[[Any], Any]] = None
        self.valid_dataset_weight: list[int] = []

        self.test_dataset: Optional[Dataset] | list[Dataset] = []
        self.test_collate_fn: Optional[Callable[[Any], Any]] = None
        self.test_dataset_weight: list[int] = []

    def build_dataset(self, path: str, use_ids: list[int], *args, **kwargs) -> Dataset:
        # So one can override the dataset creation
        return TdSegmentedExperimentDataset(path, str(Path(path) / "spec.csv"), use_ids)

    def add_dataset(self, dataset_path: str, weight: int = 1, valid_fraction: float = 0., test_fraction: float = 0.):
        if not isinstance(self.train_dataset, list):
            raise AttributeError("Module was already setup, you cannot add dataset to it.")

        if valid_fraction + test_fraction > 1:
            raise ValueError("Partitioning is invalid for given dataset")

        train_fraction = 1. - valid_fraction - test_fraction
        spec_path = Path(dataset_path) / "spec.csv"
        info = pd.read_csv(spec_path)
        ids = info["person_id"].unique()

        rng = np.random.default_rng(self.seed)
        ids = rng.permutation(ids)

        train_subjects = int(len(ids) * train_fraction)
        valid_subjects = int(len(ids) * valid_fraction)

        if train_fraction > 0:
            # Create a train dataset
            # Take the selected ids from the available
            use_ids = ids[:train_subjects]
            ids = ids[train_subjects:]
            self.train_dataset_weight.append(weight)
            self.train_dataset.append(self.build_dataset(dataset_path, use_ids.tolist()))

        if valid_fraction > 0:
            use_ids = ids[:valid_subjects]
            ids = ids[valid_subjects:]
            self.valid_dataset_weight.append(weight)
            self.valid_dataset.append(self.build_dataset(dataset_path, use_ids.tolist()))

        if test_fraction > 0 and len(ids) > 0:
            self.test_dataset_weight.append(weight)
            self.test_dataset.append(self.build_dataset(dataset_path, ids.tolist()))

    def setup(self, stage: str) -> None:
        if isinstance(self.train_dataset, list):
            self.train_dataset = ConcatDataset(self.train_dataset)
            self.valid_dataset = ConcatDataset(self.valid_dataset)
            self.test_dataset = ConcatDataset(self.test_dataset)

    def set_train_collate_fn(self, collate_fn: Callable[[Any], Any]):
        self.train_collate_fn = collate_fn

    def set_val_collate_fn(self, collate_fn: Callable[[Any], Any]):
        self.valid_collate_fn = collate_fn

    def set_test_collate_fn(self, collate_fn: Callable[[Any], Any]):
        self.test_collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_collate_fn,
            num_workers=2,
            prefetch_factor=1,
            # This is required for the way we handle shuffling
            persistent_workers=True,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.val_collate_fn,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.test_collate_fn,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=False
        )
