import pandas as pd

from core_data.extract import Extractor
from core_data.loader import DataPointsLoader
from core_data.sampler import Segmenter


class SegmentsExtractor:
    def __init__(self, base_path: str, segmenter: Segmenter, points_loader: DataPointsLoader,
                 *other_extractors: Extractor):
        self.segmenter: Segmenter = segmenter
        self.points_loader: DataPointsLoader = points_loader

        # Custom functions to be applied
        self.other_extractors = other_extractors
        self.base_path: str = base_path

        self.seen = []

    def extract_segments(self):
        for x in self.points_loader.scan():
            key = x.get_identifier()
            if key in self.seen:
                continue

            segments: list[tuple[float, float]] = self.segmenter.compute_segments(x)
            outs = [o.extract(x) for o in self.other_extractors]
            print(outs)

            df = pd.DataFrame(segments, columns=["start", "stop"])
            df.to_csv(f"{self.base_path}{x.eid}-segments.csv", index=False)
