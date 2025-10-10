import json

import torch

from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.extract import Extractor
from main.core_data.media.text import Text
from main.core_data.media.text.transforms import WhisperExtractor


class ExtractTextFromAudio(Extractor):
    def __init__(self, extractor: WhisperExtractor):
        self.extractor = extractor

    def extract(self, x: FlexibleDatasetPoint, base_path: str):
        if not hasattr(x, "txt"):
            raise ValueError("x.txt must be provided")
        txt: Text = getattr(x, "txt")

        extracted = self.extractor(torch.tensor(txt.base_audio.to_soundarray()), txt.base_audio.fps)
        output_filepath = f"{base_path}{x.eid}-txt-extract.json"
        with open(output_filepath, "w") as f:
            json.dump(extracted, f, indent=4)
        return output_filepath
