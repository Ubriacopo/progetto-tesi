from abc import ABC, abstractmethod

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

    def extract_segments(self) -> list[list[str]]:
        outs = []
        for x in self.points_loader.scan():
            key = x.get_identifier()
            if key in self.seen:
                continue

            segments: list[tuple[float, float]] = self.segmenter.compute_segments(x[EEG.modality_code()])
            local_outs = [o.extract(x, self.base_path) for o in self.other_extractors]

            df = pd.DataFrame(segments, columns=["start", "stop"])
            df.to_csv(f"{self.base_path}{x.eid}-segments.csv", index=False)
            local_outs.append(f"{self.base_path}{x.eid}-segments.csv")
            outs.append(local_outs)

        return outs
