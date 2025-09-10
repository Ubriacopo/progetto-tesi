import dataclasses
import os
from pathlib import Path
from typing import Optional

from common.data.media import Media


@dataclasses.dataclass
class EEG(Media):
    fs: float
    interval: Optional[tuple[int, int]] = None

    def export(self, base_path: str, output_path_to_relative: str = None):
        if self.data is None:
            raise ValueError("EEG data is not initialized so we cannot export")
        out_path = base_path + "_raw.fif"
        self.data.save(out_path)

        if output_path_to_relative is not None:
            self.file_path = os.path.relpath(Path(out_path).resolve(), output_path_to_relative)
        else:
            self.file_path = str(Path(out_path).resolve())

    @staticmethod
    def modality_code() -> str:
        return "eeg"
