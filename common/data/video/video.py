import dataclasses
from typing import Tuple, Optional

from common.data.media import Media


@dataclasses.dataclass
class Video(Media):
    def modality_prefix(self) -> str:
        return "vid"

    fps: int
    resolution: Tuple[int, int]
    interval: Optional[tuple[int, int]] = None
