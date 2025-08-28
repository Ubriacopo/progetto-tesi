import dataclasses
from typing import Optional

from common.data.media import Media


@dataclasses.dataclass
class EEG(Media):
    def modality_prefix(self) -> str:
        return "eeg"

    fs: float
    interval: Optional[tuple[int, int]] = None
