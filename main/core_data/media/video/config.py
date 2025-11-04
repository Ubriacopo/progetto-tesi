from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class VidTargetConfig:
    max_frames: int = 32  # Initialized on bound ViVit value


@dataclasses.dataclass
class VidSourceConfig:
    fps: int
