from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import lightning
import numpy as np
import pandas as pd
from tensordict import TensorDict
from torch.utils.data import Dataset, DataLoader

from main.utils.logging import make_logger


class BaseDatamodule(lightning.LightningDataModule, ABC):
    def __init__(self, seed: int, batch_size: int, **kwargs):
        super().__init__()
        self.logger = make_logger(self.__class__.__name__)
        self.seed: int = seed
        self.batch_size: int = batch_size

        self.train_dataset: Optional[Dataset | list[Dataset]] = None
        self.valid_dataset: Optional[Dataset | list[Dataset]] = None
        self.test_dataset: Optional[Dataset | list[Dataset]] = None
        self.is_initialized: bool = False

    def add_dataset(self, dataset_path: str, weight: int = 1, valid_fraction: float = 0., test_fraction: float = 0.):
        if self.is_initialized:
            raise AttributeError("Dataset already initialized")

        if valid_fraction + test_fraction > 1:
            raise ValueError("Partitioning is invalid for given dataset")

        train_fraction = 1. - valid_fraction - test_fraction

        spec_path = Path(dataset_path) / "spec.csv"
        ids = pd.read_csv(spec_path)["person_id"].unique()

        # Shuffle
        rng = np.random.default_rng(self.seed)
        ids = rng.permutation(ids)

        train_subjects = int(len(ids) * train_fraction)
        valid_subjects = int(len(ids) * valid_fraction)

        if train_fraction > 0:
            use_ids = ids[:train_subjects]
            ids = ids[train_subjects:]
            self.train_dataset = self.merge_with_dataset(self.train_dataset, dataset_path, use_ids.tolist(), weight)

        if valid_fraction > 0:
            use_ids = ids[:valid_subjects]
            ids = ids[valid_subjects:]
            self.valid_dataset = self.merge_with_dataset(self.valid_dataset, dataset_path, use_ids.tolist(), weight)

        if test_fraction > 0 and len(ids) > 0:
            self.test_dataset = self.merge_with_dataset(self.test_dataset, dataset_path, ids.tolist(), weight)

    def inner_initialize(self):
        self.logger.warn("Default initialization is used. Be sure that's what you want")

    def setup(self, stage: str) -> None:
        if self.is_initialized:
            return

        self.inner_initialize()
        self.is_initialized = True

    @abstractmethod
    def merge_with_dataset(self, dataset: Dataset | list[Dataset], ds_path: str, use_ids: list, weight=1):
        raise NotImplementedError("[build_dataset_element] has to be implemented.")

    @staticmethod
    @abstractmethod
    def train_collate_fn(batch: list[TensorDict]) -> TensorDict:
        raise NotImplementedError("[train_collate_fn] has to be implemented if used.")

    @staticmethod
    @abstractmethod
    def test_collate_fn(batch: list[TensorDict]) -> TensorDict:
        raise NotImplementedError("[train_collate_fn] has to be implemented if used.")

    @staticmethod
    @abstractmethod
    def valid_collate_fn(batch: list[TensorDict]) -> TensorDict:
        raise NotImplementedError("[train_collate_fn] has to be implemented if used.")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_collate_fn,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=False,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            collate_fn=self.valid_collate_fn,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.test_collate_fn,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=False
        )
