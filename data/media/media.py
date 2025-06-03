from abc import ABC, abstractmethod


class Media(ABC):
    def __init__(self, file_path: str):
        self.file_path: str = file_path
        self.loaded = False

    @abstractmethod
    def get_info(self):
        pass

    def load(self, **kwargs):
        self._inner_load(**kwargs)
        self.loaded = True

    @abstractmethod
    def _inner_load(self, **kwargs):
        pass

    def process(self, **kwargs):
        if not self.loaded:
            raise RuntimeError('Media not loaded')
        self._inner_process(**kwargs)

    @abstractmethod
    def _inner_process(self, **kwargs):
        pass
