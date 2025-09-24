from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class EegTargetConfig:
    i_max_length: int = 6
    target_fs: int = 128  # CBraMod
    cbramod_weights_path: str = ""
    max_segments: int = 10
