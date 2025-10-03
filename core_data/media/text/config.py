from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class TxtTargetConfig:
    i_max_length: int = 4
    extracted_base_path: Optional[str] = None
