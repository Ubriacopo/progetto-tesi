from abc import abstractmethod, ABC
from typing import Iterator

from main.core_data.data_point import FlexibleDatasetPoint
from main.dataset.utils import DatasetUidStore
from main.utils.logging import make_logger


class DataPointsLoader(ABC):
    """
    Loads samples of a dataset as reference points.
    """

    def __init__(self, dataset_uid_store: DatasetUidStore):
        self.dataset_uid_store: DatasetUidStore = dataset_uid_store
        self.logger = make_logger(self.__class__.__name__)

    @abstractmethod
    def scan(self) -> Iterator[FlexibleDatasetPoint]:
        pass
