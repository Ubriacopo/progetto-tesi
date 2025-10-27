from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class VidTargetConfig:
    i_max_length: int = 2  # TOOD becomes 1
    max_frames: int = 32  # Initialized on bound ViVit value


@dataclasses.dataclass
class VidSourceConfig:
    fps: int
