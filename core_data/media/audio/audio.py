import dataclasses
import os
from pathlib import Path
from typing import Optional

from core_data.media.media import Media


@dataclasses.dataclass
class Audio(Media):
    fs: float
    # Indica se questo è un segmento del Media (da quanto a quando).
    interval: Optional[tuple[int, int]] = None

    def export(self, base_path: str, output_path_to_relative: str = None):
        out_path = f"{base_path}{self.eid}.wav"
        self.data.write_audiofile(out_path)
        if output_path_to_relative is not None:
            self.file_path = os.path.relpath(Path(out_path).resolve(), output_path_to_relative)
        else:
            self.file_path = str(Path(out_path).resolve())

    @staticmethod
    def modality_code() -> str:
        return "aud"
