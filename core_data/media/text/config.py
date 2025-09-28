from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TxtTargetConfig:
    registry_store_path: str
    i_max_length: int = 4
