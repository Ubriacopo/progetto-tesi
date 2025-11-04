from __future__ import annotations

import dataclasses
from typing import Optional


@dataclasses.dataclass
class TxtTargetConfig:
    extracted_base_path: Optional[str] = None


@dataclasses.dataclass
class TxtSourceConfig:
    pass
