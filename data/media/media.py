from __future__ import annotations
from abc import ABC, abstractmethod


class Media(ABC):
    def __init__(self, file_path: str, lazy: bool = True, **kwargs):
        self.file_path: str = file_path
        self.loaded = False

        # If we are lazy we don't load else we do the steps in order.
        # If load fails we cannot process so the chain stops.
        not lazy and self.load(**kwargs) and self.process(**kwargs)

    @abstractmethod
    def get_info(self):
        pass

    def load(self, **kwargs) -> bool:
        self._inner_load(**kwargs)
        self.loaded = True
        return True

    @abstractmethod
    def _inner_load(self, **kwargs):
        pass

    def process(self, **kwargs) -> bool:
        if not self.loaded:
            raise RuntimeError('Media not loaded')
        self._inner_process(**kwargs)
        return True

    @abstractmethod
    def _inner_process(self, **kwargs):
        pass
