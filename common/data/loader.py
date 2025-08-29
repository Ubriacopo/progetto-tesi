from abc import abstractmethod, ABC
from typing import Iterator

from common.data.data_point import EEGDatasetDataPoint


class DataLoader(ABC):
    """
    Loads samples of a dataset as reference points.
    """

    @abstractmethod
    def scan(self) -> Iterator[EEGDatasetDataPoint]:
        pass
