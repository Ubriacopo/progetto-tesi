from abc import ABC, abstractmethod

from core_data.data_point import FlexibleDatasetPoint


class Extractor(ABC):
    @abstractmethod
    def extract(self, x: FlexibleDatasetPoint) -> str:
        pass
