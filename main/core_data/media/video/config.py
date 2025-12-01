from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class VidTargetConfig:
    max_frames: int = 32  # Initialized on bound ViVit value
    target_fps: int = 25

@dataclasses.dataclass
class VidSourceConfig:
    fps: int
