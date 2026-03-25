from pathlib import Path
from typing import Optional

import lightning
import numpy as np
import pandas as pd
import tensordict
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from tensordict import TensorDict
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from main.core_data.ds.td_dataset import TdSegmentedExperimentDataset
from main.utils.logging import make_logger


# todo probabilemnte dovro usare ancorea h5py
class LinearProbeDataModule(lightning.LightningDataModule):
    @staticmethod
    def collate_fn(batch: list[TensorDict]):
        batch = [b.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )[:15] for b in batch]
        return tensordict.pad_sequence(batch, 0, return_mask="pad_mask")

    def __init__(self, seed: int, batch_size: int):
        super().__init__()
        self.logger = make_logger(self.__class__.name)
        self.train_dataset: Optional[Dataset] | list[Dataset] = []
        self.train_dataset_weight: list[int] = []

        self.valid_dataset: Optional[Dataset] | list[Dataset] = []
        self.valid_dataset_weight: list[int] = []

        self.test_dataset: Optional[Dataset] | list[Dataset] = []
        self.test_dataset_weight: list[int] = []

        self.seed = seed
        self.dataset_paths: list[str] = []
        self.batch_size = batch_size

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
            self.train_dataset.append(TdSegmentedExperimentDataset(dataset_path, str(spec_path), use_ids.tolist()))

        if valid_fraction > 0:
            use_ids = ids[:valid_subjects]
            ids = ids[valid_subjects:]
            self.valid_dataset_weight.append(weight)
            self.valid_dataset.append(TdSegmentedExperimentDataset(dataset_path, str(spec_path), use_ids.tolist()))

        if test_fraction > 0 and len(ids) > 0:
            self.test_dataset_weight.append(weight)
            self.test_dataset.append(TdSegmentedExperimentDataset(dataset_path, str(spec_path), ids.tolist()))

    def setup(self, stage: str) -> None:
        # TODO Custom dataset that handles weigth. For first probe this is not necessary
        self.train_dataset = ConcatDataset(self.train_dataset)
        self.valid_dataset = ConcatDataset(self.valid_dataset)
        self.test_dataset = ConcatDataset(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=2,
            prefetch_factor=1,
            # This is required for the way we handle shuffling
            persistent_workers=True,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=False
        )
