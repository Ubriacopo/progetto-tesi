from typing import Optional

import lightning
import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader, IterableDataset

from main.core_data.dataset import MultiDataset, \
    CachableDatasetDescriptor, H5KdSourceDataset, RoundRobinMultiDataset, H5KdDataset


class KdTrainDataModule(lightning.LightningDataModule):
    def __init__(self,
                 dataset_paths: list[CachableDatasetDescriptor],
                 # Parameters for stuff less related to the data itself
                 batch_size: int, batches_per_epoch: int,
                 seed: int, collate_fn=lambda x: torch.stack(x, dim=0)):
        """
        :param student_keys: list of keys that appear in the student records as tensor_dicts
        :param teacher_keys: list of keys that appear in the teacher records as tensor_dicts
        :param dataset_paths: list of zipped dataset paths of matching student and teacher
        ... todo
        """
        super().__init__()
        self.shards_path: list[CachableDatasetDescriptor] = dataset_paths

        self.train_dataset: Optional[IterableDataset] = None
        self.valid_dataset: Optional[MultiDataset] = None
        self.test_dataset: Optional[MultiDataset] = None

        # Tunable settings to make training faster. Non data related
        self.seed = seed
        self.collate_fn = collate_fn
        self.batch_size: int = batch_size
        self.batches_per_epoch: int = batches_per_epoch

    def setup(self, stage: str) -> None:
        datasets, weights = [], []
        for shards_path in self.shards_path:
            datasets.append(
                H5KdDataset(dataset_path=shards_path.dataset_path, prefix="train")
            )
            weights.append(shards_path.dataset_weight)

        ds = RoundRobinMultiDataset(datasets, weights, seed=self.seed)
        self.train_dataset = ds

    def _move(self, x, device):
        import torch
        if isinstance(x, torch.Tensor):
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
        fn = self.collate_fn
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=fn,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True
        )
