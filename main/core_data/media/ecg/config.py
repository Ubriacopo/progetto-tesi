from __future__ import annotations

import dataclasses
from abc import abstractmethod, ABC
from typing import Callable

from hydra.utils import get_object

from main.core_data.media.ecg import ECG


@dataclasses.dataclass
class EcgTargetConfig:
    fs: int = 128
    fm_endpoint: str = "localhost:7860/extract_features"


@dataclasses.dataclass
class EcgSourceConfig(ABC):
    LEAD_NAMES: list[str]

    @staticmethod
    @abstractmethod
    def prepare_ecg(ecg: ECG) -> ECG:
        pass
