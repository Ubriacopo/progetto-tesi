import dataclasses
from typing import Optional

from common.data.media import Media


@dataclasses.dataclass
class EEG(Media):
    def export(self, base_path: str = None):
        if self.data is None:
            raise ValueError("EEG data is not initialized so we cannot export")
        self.data.save(base_path + "_raw.fif")

    @staticmethod
    def modality_code() -> str:
        return "eeg"

    fs: float
    interval: Optional[tuple[int, int]] = None
