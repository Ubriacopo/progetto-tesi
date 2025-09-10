import dataclasses
import os
from pathlib import Path
from typing import Tuple, Optional

from common.data.media import Media


@dataclasses.dataclass
class Video(Media):
    def export(self, base_path: str, output_path_to_relative: str = None):
        out_path = f"{base_path}{self.eid}.mp4"
        self.data.write_videofile(out_path, audio=False, codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"], )
        if output_path_to_relative is not None:
            self.file_path = os.path.relpath(Path(out_path).resolve(), output_path_to_relative)
        else:
            self.file_path = str(Path(out_path).resolve())

    @staticmethod
    def modality_code() -> str:
        return "vid"

    fps: int
    resolution: Tuple[int, int]
    interval: Optional[tuple[int, int]] = None
