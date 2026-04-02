import torch
from tensordict import TensorDict, tensordict

from main.core_data.ds.td_dataset import TdSegmentedExperimentDataset
from main.model.downstream.core.datamodule import BaseDatamodule


class ClareDataModule(BaseDatamodule):
    def __init__(self, dataset_path: str, seed: int, batch_size: int):
        super().__init__(seed=seed, batch_size=batch_size)
        self.add_dataset(dataset_path, valid_fraction=0.1, test_fraction=0.15)

    def merge_with_dataset(self, dataset, ds_path: str, use_ids: list, weight=1):
        return TdSegmentedExperimentDataset(ds_path, ds_path + "/spec.csv", use_ids)

    @staticmethod
    def train_collate_fn(batch: list[TensorDict]) -> TensorDict:
        # TODO
        return_object = []
        for td in batch:
            # Video indexes are scaled from [1-28], labels from [0-27]
            # Pool labels in even
            one_hot = td["assessment", "scores"][0]

            cls = one_hot.argmax(dim=-1).long()
            emotion_label = cls // 2
            speaking = cls % 2

            td["assessment", "score"] = torch.full(td.batch_size, emotion_label.item(), dtype=torch.long)
            td["assessment", "speaking"] = torch.full(td.batch_size, speaking.item(), dtype=torch.long)
            td = td.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )
            return_object.append(td)

        return tensordict.pad_sequence(return_object, 0, return_mask="pad_mask")

    @staticmethod
    def test_collate_fn(batch: list[TensorDict]) -> TensorDict:
        return ClareDataModule.train_collate_fn(batch)

    @staticmethod
    def valid_collate_fn(batch: list[TensorDict]) -> TensorDict:
        return ClareDataModule.train_collate_fn(batch)
