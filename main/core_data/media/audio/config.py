from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class AudTargetConfig:
    fs: int = 16000


@dataclasses.dataclass
class AudSourceConfig:
    fs: int
