import dataclasses
import os
from pathlib import Path
from typing import Optional

from moviepy import AudioFileClip

from core_data.media.media import Media


@dataclasses.dataclass
class Text(Media):
    base_audio: Optional[AudioFileClip] = None
    text_context: Optional[dict] = None
    interval: Optional[tuple[int, int]] = None

    def export(self, base_path: str, output_path_to_relative: str = None):
        out_path = base_path + f'{self.eid}.txt'
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self.data)
        if output_path_to_relative is not None:
            self.file_path = os.path.relpath(Path(out_path).resolve(), output_path_to_relative)
        else:
            self.file_path = str(Path(out_path).resolve())

    @staticmethod
    def modality_code() -> str:
        return "txt"
