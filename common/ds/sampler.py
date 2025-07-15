from abc import abstractmethod, ABC


class DataSampler(ABC):
    @abstractmethod
    def process_sample(self, sample):
        pass
