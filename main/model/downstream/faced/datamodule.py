from typing import Callable

import torch
from tensordict import TensorDict, tensordict

from main.core_data.ds.td_dataset import TdSegmentedExperimentDataset
from main.dataset.faced.label_map import LabelMap
from main.model.downstream.base import BaseDatamodule


class FacedDataModule(BaseDatamodule):
    def __init__(self, faced_path: str, seed: int, batch_size: int):
        super().__init__(seed=seed, batch_size=batch_size)
        self.add_dataset(faced_path, valid_fraction=0.1, test_fraction=0.15)

    def merge_with_dataset(self, dataset, ds_path: str, use_ids: list, weight=1):
        return TdSegmentedExperimentDataset(ds_path, ds_path + "/spec.csv", use_ids)

    @staticmethod
    def train_collate_fn(batch: list[TensorDict]) -> Callable[[list[TensorDict]], TensorDict]:
        # TODO Verifica
        return_object = []
        for td in batch:
            # Video indexes are scaled from [1-28], labels from [0-27]
            label = LabelMap.num_labels[td["meta", "trial"][0].item() - 1]
            td["assessment", "score"] = torch.full(td.batch_size, label, dtype=torch.long)
            td = td.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )
            return_object.append(td)
        return tensordict.pad_sequence(return_object, 0, return_mask="pad_mask")

    @staticmethod
    def test_collate_fn(batch: list[TensorDict]) -> Callable:
        return FacedDataModule.train_collate_fn(batch)

    @staticmethod
    def valid_collate_fn(batch: list[TensorDict]) -> Callable:
        return FacedDataModule.train_collate_fn(batch)