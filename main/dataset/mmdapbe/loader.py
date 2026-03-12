from typing import Iterator

from scipy.io import loadmat

from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.loader import DataPointsLoader
from main.dataset.dreamer.config import DreamerConfig
from main.dataset.mmdapbe.config import MmdapbeConfig
from main.dataset.utils import DatasetUidStore


class MmdapbePointsLoader(DataPointsLoader):
    DATASET_ID: int = 5

    def __init__(self, base_path: str, dataset_uid_store: DatasetUidStore, config: MmdapbeConfig = MmdapbeConfig()):
        super().__init__(dataset_uid_store)
        self.base_path = base_path
        self.config: DreamerConfig = config

    def __len__(self) -> int:
        if self.length is not None:
            return self.length

    # TODO finish
    def scan(self) -> Iterator[FlexibleDatasetPoint]:
        yield FlexibleDatasetPoint()
