from typing import Optional, Literal

import lightning
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader, IterableDataset, ChainDataset

from main.core_data.dataset import CacheableDatasetDescriptor, RoundRobinBatchMultiDataset, H5KdDataset
from main.utils.logging import make_logger


def collate(x):
    return x


class KdTrainDataModule(lightning.LightningDataModule):
    def __init__(
            self,
            dataset_paths: list[CacheableDatasetDescriptor],
            # Parameters for stuff less related to the data itself
            batch_size: int,
            seed: int,
            dequantize_keys: list[str],
            collate_fn=collate,
            restore_iteration: int = None,
            # These are only here for hp tuning at the moment
            train_fraction: float = None,
            valid_fraction: float = None,
            test_fraction: float = None,
            take_keys: list[str] = (),
    ):
        """
        :param student_keys: list of keys that appear in the student records as tensor_dicts
        :param teacher_keys: list of keys that appear in the teacher records as tensor_dicts
        :param dataset_paths: list of zipped dataset paths of matching student and teacher
        ... todo
        """
        super().__init__()
        self.logger = make_logger(self.__class__.name)
        self.shards_path: list[CacheableDatasetDescriptor] = dataset_paths

        self.train_dataset: Optional[IterableDataset] = None
        self.valid_dataset: Optional[IterableDataset] = None

        self.test_ds_collection: dict[str, IterableDataset] = {}
        self.test_dataset: Optional[IterableDataset] = None

        # Tunable settings to make training faster. Non data related
        self.seed = seed
        self.collate_fn = collate_fn
        self.batch_size: int = batch_size
        self.restore_iteration: int = restore_iteration

        self.load_test = True
        self._dequantize_keys: list[str] = dequantize_keys

        self.setup_done: bool = False
        self.lengths: dict[str, int] = {"train": 0, "val": 0, "test": 0}

        self.train_fraction: Optional[float] = train_fraction

        self.valid_fraction: Optional[float] = valid_fraction
        self.test_fraction: Optional[float] = test_fraction
        if self.valid_fraction is not None or self.test_fraction is not None:
            self.logger.warning("You have set fraction on test/validation sets. Beware of unwanted behaviours")

        self.take_keys = take_keys

    def dequantize_keys(self) -> list[str]:
        return self._dequantize_keys

    def size(self, split: Literal["train", "val", "test"]) -> int:
        if not self.setup_done:
            raise ValueError("Setup has to be done before calling length")
        return self.lengths[split]

    def setup(self, stage: str) -> None:

        datasets, weights = [], []
        val_datasets, test_datasets = [], []

        it_id = self.restore_iteration
        for shards_path in self.shards_path:
            ds_path = shards_path.dataset_path

            weights.append(shards_path.dataset_weight)

            # Can't handle 12 on b=128 so we reduce it. Hacky.
            factor = 8 if self.batch_size > 64 else 12

            dataset = H5KdDataset(
                ds_path,
                prefix="train",
                block_size=32,
                buffer_size=self.batch_size * factor,
                batch_size=self.batch_size,
                iterator_id=it_id,
                limit_data=self.train_fraction,
                take_keys=self.take_keys
            )

            train_samples = len(dataset)
            self.lengths["train"] += train_samples
            self.logger.debug(f"Dataset with f{shards_path} has a total of {train_samples} train samples")

            datasets.append(dataset)

            val_dataset = H5KdDataset(
                ds_path, prefix="val",
                block_size=32,
                buffer_size=96,
                batch_size=self.batch_size,
                shuffle=False,
                limit_data=self.valid_fraction,
                take_keys=self.take_keys
            )

            val_samples = len(val_dataset)
            self.lengths["val"] += val_samples
            self.logger.debug(f"Dataset with f{shards_path} has a total of {val_samples} validation samples")
            val_datasets.append(val_dataset)

            test_samples = 0
            if self.load_test:
                test_ds = H5KdDataset(
                    ds_path,
                    prefix="test",
                    block_size=32,
                    buffer_size=96,
                    batch_size=self.batch_size,
                    limit_data=self.test_fraction,
                    take_keys=self.take_keys
                )

                test_samples = len(test_ds)
                self.lengths["test"] += test_samples
                self.logger.debug(f"Dataset with f{shards_path} has a total of {test_samples} validation samples")
                self.test_ds_collection[shards_path.dataset_path] = test_ds

            self.logger.debug(
                f"Dataset with f{shards_path} has a total of {train_samples + val_samples + test_samples} validation samples"
            )

        self.train_dataset = RoundRobinBatchMultiDataset(datasets, weights, seed=self.seed, consecutive_batches=8)
        self.valid_dataset = ChainDataset(val_datasets)
        self.test_dataset = ChainDataset(self.test_ds_collection.values())

        self.setup_done = True

    def _move(self, x, device):
        if isinstance(x, torch.Tensor) or isinstance(x, TensorDict):
            return x.to(device, non_blocking=True)
        if isinstance(x, dict):
            return {k: self._move(v, device) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            t = [self._move(v, device) for v in x]
            return type(x)(t)
        return x  # leave non-tensors alone

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return self._move(batch, device)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            collate_fn=self.collate_fn,
            num_workers=2,
            prefetch_factor=1,
            # This is required for the way we handle shuffling
            persistent_workers=True,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=None,
            collate_fn=self.collate_fn,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=None,
            collate_fn=self.collate_fn,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=False
        )

    def test_for_ds(self):
        return {
            name: DataLoader(
                ds,
                batch_size=None,
                collate_fn=self.collate_fn,
                num_workers=1,
                prefetch_factor=1,
                persistent_workers=True,
                pin_memory=False
            )
            for name, ds in self.test_ds_collection.items()
        }
