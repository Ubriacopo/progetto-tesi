from typing import Optional

import lightning
import tensordict
import torch
from torch.utils.data import DataLoader

from main.core_data.dataset import RequiredKey, MultiDataset, \
    SequentialPerDatasetBatchSampler, CachableDatasetDescriptor, H5KdSourceDataset


def olddefault_collate_fn(batch):
    return tensordict.stack(batch)


def default_collate_fn(batch):
    out = {"student": {}, "teacher": {}}
    for who in ("student", "teacher"):
        for key in batch[0][who].keys():
            out[who][key] = {
                "data": torch.stack([b[who][key]["data"] for b in batch], 0),
                "mask": torch.stack([b[who][key]["mask"] for b in batch], 0),
                "scales": torch.stack([b[who][key]["scales"] for b in batch], 0),
            }
    return out


class KdTrainDataModule(lightning.LightningDataModule):
    def __init__(self,
                 student_keys: list[RequiredKey], teacher_keys: list[RequiredKey],
                 dataset_paths: list[CachableDatasetDescriptor],
                 # Parameters for stuff less related to the data itself
                 batch_size: int, batches_per_epoch: int,
                 seed: int, collate_fn=default_collate_fn):
        """
        :param student_keys: list of keys that appear in the student records as tensor_dicts
        :param teacher_keys: list of keys that appear in the teacher records as tensor_dicts
        :param dataset_paths: list of zipped dataset paths of matching student and teacher
        ... todo
        """
        super().__init__()
        self.student_keys: list[RequiredKey] = student_keys
        self.teacher_keys: list[RequiredKey] = teacher_keys
        self.shards_path: list[CachableDatasetDescriptor] = dataset_paths

        self.train_dataset: Optional[MultiDataset] = None
        self.valid_dataset: Optional[MultiDataset] = None
        self.test_dataset: Optional[MultiDataset] = None

        # Tunable settings to make training faster. Non data related
        self.seed = seed
        self.collate_fn = collate_fn
        self.batch_size: int = batch_size
        self.batches_per_epoch: int = batches_per_epoch

    def setup(self, stage: str) -> None:
        dataset_pairs = []
        for shards_path in self.shards_path:
            dataset_pairs.append(H5KdSourceDataset(shards_path.dataset_spec_file))

        ds = MultiDataset(dataset_pairs)
        # todo non penso di poterlo fare
        self.train_dataset, self.valid_dataset, self.test_dataset = ds.split(0.75, 0.15, seed=self.seed)

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

    # def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
    #    return batch.to(device, non_blocking=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            num_workers=2,
            prefetch_factor=2,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_sampler=SequentialPerDatasetBatchSampler(multi=self.valid_dataset, batch_size=self.batch_size, ),
            collate_fn=self.collate_fn,
            num_workers=0,
            prefetch_factor=None,
            persistent_workers=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_sampler=SequentialPerDatasetBatchSampler(multi=self.test_dataset, batch_size=self.batch_size, ),
            collate_fn=self.collate_fn,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=True
        )
