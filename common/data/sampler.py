import dataclasses
from abc import abstractmethod, ABC


@dataclasses.dataclass
class SamplingDescriptor:
    sample_index: int
    media_path: str
    original_filename: str
    split_index: int
    start_timestamp: int
    stop_timestamp: int
    experiment_id: str
    data_file: str
    data_index: int

class DataSampler(ABC):
    @abstractmethod
    def process_sample(self, sample):
        pass
