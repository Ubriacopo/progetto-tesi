import dataclasses
import os
from pathlib import Path

from common.data.media import Media


@dataclasses.dataclass
class Text(Media):
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
