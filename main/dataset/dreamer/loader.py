from pathlib import Path
from typing import Iterator

from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.loader import DataPointsLoader
from main.dataset.utils import DatasetUidStore


class DreamerPointsLoader(DataPointsLoader):
    def __init__(self, base_path: str, dataset_uid_store: DatasetUidStore):
        super().__init__(dataset_uid_store)
        self.base_path = base_path

    # TODO finish
    def scan(self) -> Iterator[FlexibleDatasetPoint]:
        processed_data = Path(self.base_path + "data_preprocessed_python/")

        yield FlexibleDatasetPoint()
