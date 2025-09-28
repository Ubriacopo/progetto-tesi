from abc import abstractmethod, ABC
from typing import Iterator

from core_data.data_point import FlexibleDatasetPoint


class DataPointsLoader(ABC):
    """
    Loads samples of a dataset as reference points.
    """

    @abstractmethod
    def scan(self) -> Iterator[FlexibleDatasetPoint]:
        pass
