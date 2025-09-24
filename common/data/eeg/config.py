from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class EegTargetConfig:
    cbramod_weights_path: str
    i_max_length: int = 6
    target_fs: int = 200  # CBraMod
    max_segments: int = 10 # From CBraMod
