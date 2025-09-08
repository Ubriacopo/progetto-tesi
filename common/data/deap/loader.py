from typing import Iterator

from common.data.data_point import EEGDatasetDataPoint
from common.data.loader import DataPointsLoader


class DeapPointsLoader(DataPointsLoader):
    def __init__(self, base_path: str):
        super().__init__()
        self.base_path = base_path

    def scan(self) -> Iterator[EEGDatasetDataPoint]:
        pass
