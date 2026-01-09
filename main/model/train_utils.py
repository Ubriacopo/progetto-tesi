from typing import Optional, Any

import lightning
import tensordict
import torch
from torch.utils.data import DataLoader

from main.core_data.dataset import RequiredKey, FlexibleEmbeddingsSpecMediaDatasetSlow, MultiDataset, \
    MultiDatasetQueueBatchSampler, SequentialPerDatasetBatchSampler
from main.model.kd_dataset_wrapper import KdDatasetWrapper


def default_collate_fn(batch):
    td = tensordict.stack(batch)
    td.pop(("student", "meta"))
    td.pop(("teacher", "meta"))
    # For performance reasons
    td["student", "vid", "data"] = td["student", "vid", "data"].to(torch.float16)
    return td


class KdTrainDataModule(lightning.LightningDataModule):
    def __init__(self,
                 student_keys: list[RequiredKey], teacher_keys: list[RequiredKey],
                 dataset_paths: list[tuple[str, str]], student_pivot: str, teacher_pivot: str,
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
        self.dataset_paths: list[tuple[str, str]] = dataset_paths
        self.student_pivot: str = student_pivot
        self.teacher_pivot: str = teacher_pivot

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
        for student_ds_path, teacher_ds_path in self.dataset_paths:
            dataset_pairs.append(
                KdDatasetWrapper(
                    student=FlexibleEmbeddingsSpecMediaDatasetSlow(
                        student_ds_path, self.student_keys, main_key=self.student_pivot
                    ),
                    teacher=FlexibleEmbeddingsSpecMediaDatasetSlow(
                        teacher_ds_path, self.teacher_keys, main_key=self.teacher_pivot, squeeze_mask=True
                    )
                )
            )
        ds = MultiDataset(dataset_pairs)
        self.train_dataset, self.valid_dataset, self.test_dataset = ds.split(0.75, 0.15, seed=self.seed)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        return batch.to(device, non_blocking=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_sampler=MultiDatasetQueueBatchSampler(
                multi=self.train_dataset,
                batch_size=self.batch_size,
                batches_per_epoch=self.batches_per_epoch,  # To allow small ds to be present and not too repetitive
                alpha=0.0,  # Probability normalization or stuff like that on the draws from different ds based on size
                generator=torch.Generator().manual_seed(self.seed)
            ),
            collate_fn=self.collate_fn,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_sampler=SequentialPerDatasetBatchSampler(multi=self.valid_dataset, batch_size=self.batch_size, ),
            collate_fn=self.collate_fn,
            num_workers=4,
            prefetch_factor=2,
            persistent_workers=True
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
