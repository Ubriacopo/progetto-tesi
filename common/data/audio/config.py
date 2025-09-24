from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class AudTargetConfig:
    i_max_length: int = 4
    fs: int = 16000
