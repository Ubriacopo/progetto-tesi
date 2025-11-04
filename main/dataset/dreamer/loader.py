from pathlib import Path
from typing import Iterator

from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.loader import DataPointsLoader


class DreamerPointsLoader(DataPointsLoader):
    def __init__(self, base_path: str):
        super().__init__()
        self.base_path = base_path

    # TODO finish
    def scan(self) -> Iterator[FlexibleDatasetPoint]:
        processed_data = Path(self.base_path + "data_preprocessed_python/")

        yield FlexibleDatasetPoint()
