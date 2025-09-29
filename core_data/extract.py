from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from core_data.data_point import FlexibleDatasetPoint
from core_data.loader import DataPointsLoader
from core_data.media.eeg import EEG
from core_data.sampler import Segmenter


class Extractor(ABC):
    @abstractmethod
    def extract(self, x: FlexibleDatasetPoint, output_path: str) -> str:
        pass


class SegmentBasedExtractionProcessor:
    def __init__(self, *other_extractors: Extractor, base_path: str, segmenter: Segmenter, loader: DataPointsLoader):
        self.segmenter: Segmenter = segmenter
        self.points_loader: DataPointsLoader = loader
        # Custom functions to be applied
        self.other_extractors: tuple[Extractor, ...] = other_extractors
        self.base_path: str = base_path

        self.seen = []
        Path(self.base_path).mkdir(parents=True, exist_ok=True)
        for file in Path(self.base_path).glob("*-segments.csv"):
            self.seen.append(str(file.name).replace("-segments.csv", ""))

    def extract_segments(self) -> list[list[str]]:
        outs = []
        for x in self.points_loader.scan():
            key = x.get_identifier()
            if x[key] in self.seen:
                print(f"Skipping {x[key]} as it was already extracted")
                continue

            segments: list[tuple[float, float]] = self.segmenter.compute_segments(x[EEG.modality_code()])
            local_outs = [o.extract(x, self.base_path) for o in self.other_extractors]

            df = pd.DataFrame(segments, columns=["start", "stop"])
            df.to_csv(f"{self.base_path}{x.eid}-segments.csv", index=False)
            local_outs.append(f"{self.base_path}{x.eid}-segments.csv")
            outs.append(local_outs)

        return outs
