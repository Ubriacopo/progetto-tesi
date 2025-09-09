import dataclasses
from typing import Optional

from common.data.media import Media


@dataclasses.dataclass
class EEG(Media):
    @staticmethod
    def modality_code() -> str:
        return "eeg"

    fs: float
    interval: Optional[tuple[int, int]] = None
