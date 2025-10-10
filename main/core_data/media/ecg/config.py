from __future__ import annotations

import dataclasses
from typing import Callable

from hydra.utils import get_object


@dataclasses.dataclass
class EcgHydraConfig:
    use: bool = False
    prepare: str = ""
    i_max_length: int = 4
    target_fs: int = 128
    fm_endpoint: str = "localhost:7860/extract_features"

    def get_prepare(self):
        return get_object(self.prepare)


@dataclasses.dataclass
class EcgTargetConfig:
    prepare: Callable
    i_max_length: int = 4
    target_fs: int = 128
    fm_endpoint: str = "localhost:7860/extract_features"

    @staticmethod
    def from_hydra(cfg: EcgHydraConfig) -> EcgTargetConfig:
        prepare = get_object(cfg.prepare)
        filtered = {k: v for k, v in cfg.items() if k not in ("prepare", "use")}
        return EcgTargetConfig(prepare=prepare, **filtered)
