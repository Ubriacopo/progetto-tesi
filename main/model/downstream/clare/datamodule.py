from tensordict import TensorDict

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
        return_object = []
        for td in batch:
            if td["eeg"].shape == (1,):
                # 13 samples are broken in train, 1 in test
                continue

            td = td.exclude("meta", ("assessment", "scales"), ("assessment", "labels"), )
            return_object.append(td)

        return TensorDict.stack(return_object, 0)

    @staticmethod
    def test_collate_fn(batch: list[TensorDict]) -> TensorDict:
        return ClareDataModule.train_collate_fn(batch)

    @staticmethod
    def valid_collate_fn(batch: list[TensorDict]) -> TensorDict:
        return ClareDataModule.train_collate_fn(batch)
