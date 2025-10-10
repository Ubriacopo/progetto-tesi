from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class AudTargetConfig:
    i_max_length: float = 0.96 # This comes from standard VGG makes it simpler to be compatible with Audio FMs
    fs: int = 16000
