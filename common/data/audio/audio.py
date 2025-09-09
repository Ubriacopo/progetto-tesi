import dataclasses
from typing import Optional

from common.data.media import Media


@dataclasses.dataclass
class Audio(Media):
    @staticmethod
    def modality_code() -> str:
        return "aud"

    fs: float
    # Indica se questo è un segmento del Media (da quanto a quando).
    interval: Optional[tuple[int, int]] = None
