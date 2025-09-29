import json

from core_data.data_point import FlexibleDatasetPoint
from core_data.extract import Extractor
from core_data.media.text import Text
from core_data.media.text.transforms import WhisperExtractor


class ExtractTextFromAudio(Extractor):
    def __init__(self, extractor: WhisperExtractor, base_path: str):
        self.extractor = extractor
        self.base_path: str = base_path

    def extract(self, x: FlexibleDatasetPoint):
        if not hasattr(x, "txt"):
            raise ValueError("x.txt must be provided")
        txt: Text = getattr(x, "txt")

        extracted = self.extractor(txt.base_audio.to_soundarray(), txt.base_audio.fps)
        output_filepath = f"{self.base_path}{x.eid}-txt-extract.json"
        with open(output_filepath, "w") as f:
            json.dump(extracted, f, indent=4)
        return output_filepath
