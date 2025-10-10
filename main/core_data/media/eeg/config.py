from __future__ import annotations

import dataclasses

from omegaconf import MISSING


@dataclasses.dataclass
class EegTargetConfig:
    cbramod_weights_path: str = MISSING
    target_fs: int = 200  # CBraMod
    max_segments: int = 10  # From CBraMod


# TODO Uniformando sarà più facile?
@dataclasses.dataclass
class EegSourceConfig:
    pass
