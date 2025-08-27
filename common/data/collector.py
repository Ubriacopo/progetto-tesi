import dataclasses
from abc import abstractmethod, ABC


@dataclasses.dataclass
class DatasetDataCollection:
    entry_id: str
    # eeg_data: list[np.ndarray] | np.ndarray
    # mediafile_path: str | Path  # Extensionless.


class DataCollector(ABC):
    def __init__(self):
        """
        Collects info of each entry of our dataset.
        """
        self.data: list[DatasetDataCollection] = []
        self.scanned: bool = False

    @abstractmethod
    def _scan(self, *args, **kwargs) -> list[DatasetDataCollection]:
        pass

    def scan(self, force: bool = False, *args, **kwargs) -> list[DatasetDataCollection]:
        if self.scanned and not force:
            return self.data
        return self._scan(*args, **kwargs)
