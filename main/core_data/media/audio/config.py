from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class AudTargetConfig:
    # i_max_length: float = 0.96 # This comes from standard VGG makes it simpler to be compatible with Audio FMs
    i_max_length: float = 1  # Match the EEG building block size
    fs: int = 16000
