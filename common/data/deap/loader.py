from pathlib import Path
from typing import Iterator

import numpy as np

from common.data.data_point import EEGDatasetDataPoint
from common.data.loader import DataPointsLoader


class DeapPointsLoader(DataPointsLoader):
    def __init__(self, base_path: str):
        super().__init__()
        self.base_path = base_path

    def scan(self) -> Iterator[EEGDatasetDataPoint]:
        processed_data = Path(self.base_path + "data_preprocessed_python/")

        for i in processed_data.iterdir():
            if i.suffix != ".dat":
                continue
            # Contiene:
            #   - labels (40, 4): Autovalutazioni (o valutazioni controlla) per Valence-Arousal-Dominance-Liking
            #   - data (128Hz) (40, 40, 8064) (vid x channel x data): I dati EEG. Per la mappa facciamo affidamento al sito.
            data = np.load(i, allow_pickle=True, encoding="latin1")
            uid = i.stem

            # In pre-processing forse per EEG facciamo poco (se non embeddings direttamente).

            yield EEGDatasetDataPoint()
