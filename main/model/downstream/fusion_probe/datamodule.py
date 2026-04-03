import tensordict
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset, ConcatDataset

from main.core_data.ds.td_dataset import TdSegmentedExperimentDataset
from main.model.downstream.base import BaseDatamodule


class FusionDataModule(BaseDatamodule):
    def __init__(self, seed: int, batch_size: int):
        super().__init__(seed=seed, batch_size=batch_size)

    @staticmethod
    def train_collate_fn(batch: list[TensorDict]) -> TensorDict:
        for td in batch:
            # Only take the first 5 (V/A/D/L/F) because the train dataset (AMIGOS) has more
            td["assessment", "scores"] = td["assessment", "scores"][:, :3]
            if td["meta", "dataset_id"][0].item() == 3:
                # Missmapped DEAP dataset
                td["assessment", "scores"] = td["assessment", "scores"][:, [1, 0, 2]]

            # Add missing tensors
            if "vid" not in td:
                td["vid"] = TensorDict({
                    "data": torch.zeros((td.shape[0], 8, 256, 768), dtype=torch.int8),
                    "mask": torch.zeros((td.shape[0], 8), dtype=torch.bool),
                    "scales": torch.zeros((td.shape[0], 8, 256, 1), dtype=torch.float16)
                })

            if "aud" not in td:
                td["aud"] = TensorDict({
                    "data": torch.zeros((td.shape[0], 8, 199, 768), dtype=torch.int8),
                    "mask": torch.zeros((td.shape[0], 8), dtype=torch.bool),
                    "scales": torch.zeros((td.shape[0], 8, 199, 1), dtype=torch.float16)
                })

            if "ecg" not in td:
                td["ecg"] = TensorDict({
                    "data": torch.zeros((td.shape[0], 8, 32, 256), dtype=torch.int8),
                    "mask": torch.zeros((td.shape[0], 8), dtype=torch.bool),
                    "scales": torch.zeros((td.shape[0], 8, 32, 1), dtype=torch.float16)
                })

        batch = [b.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )[:10] for b in batch]
        return tensordict.pad_sequence(batch, 0, return_mask="pad_mask")

    @staticmethod
    def test_collate_fn(batch: list[TensorDict]) -> TensorDict:
        return FusionDataModule.train_collate_fn(batch)

        # todo resort the labels of DEAP
        for td in batch:
            # Only take the first 5 (V/A/D/L/F) because the train dataset (AMIGOS) has more
            td["assessment", "scores"] = td["assessment", "scores"][:, :3]

        batch = [b.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )[:10] for b in batch]
        return tensordict.pad_sequence(batch, 0, return_mask="pad_mask")

    @staticmethod
    def valid_collate_fn(batch: list[TensorDict]) -> TensorDict:
        return FusionDataModule.train_collate_fn(batch)

    def merge_with_dataset(self, dataset: Dataset | list[Dataset], ds_path: str, use_ids: list, weight=1):
        ds = TdSegmentedExperimentDataset(ds_path, ds_path + "/spec.csv", use_ids)
        if dataset is not None and isinstance(dataset, list):
            # Append to existing dataset
            dataset.append(ds)
            # Return the updated dataset
            return dataset
        # No instance was created so we make it a list
        return [ds]

    def inner_initialize(self):
        self.train_dataset = ConcatDataset(self.train_dataset) if len(self.train_dataset) > 1 else self.train_dataset[0]
        self.valid_dataset = ConcatDataset(self.valid_dataset) if len(self.valid_dataset) > 1 else self.valid_dataset[0]
        self.test_dataset = ConcatDataset(self.test_dataset) if len(self.test_dataset) > 1 else self.test_dataset[0]

    def setup(self, stage: str) -> None:
        if self.is_initialized:
            return
        self.inner_initialize()
        self.is_initialized = True
