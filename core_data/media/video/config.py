from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class VidTargetConfig:
    i_max_length: int = 2
    max_fps: int = 32  # Initialized on bound ViVit value
