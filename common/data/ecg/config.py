from __future__ import annotations

import dataclasses
from typing import Callable


@dataclasses.dataclass
class EcgTargetConfig:
    prepare: Callable
    i_max_length: int = 4
    target_fs: int = 128
    fm_endpoint: str = "localhost:7860/extract_features"
