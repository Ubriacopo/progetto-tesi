import dataclasses
from typing import Optional

from common.data.media import Media


@dataclasses.dataclass
class Audio(Media):
    def modality_prefix(self) -> str:
        return "aud"

    fs: float
    # Indica se questo è un segmento del Media (da quanto a quando).
    interval: Optional[tuple[int, int]] = None
