from pathlib import Path
from typing import Iterator

import numpy as np
from moviepy import VideoFileClip

from core_data.data_point import FlexibleDatasetPoint
from core_data.media.eeg import EEG
from core_data.loader import DataPointsLoader
from core_data.media.video import Video


class DeapPointsLoader(DataPointsLoader):
    def __init__(self, base_path: str):
        super().__init__()
        self.base_path = base_path

    def scan(self) -> Iterator[FlexibleDatasetPoint]:
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
            for idx, (labels, trial) in enumerate(zip(data["labels"], data["data"])):
                eid: str = f"{uid}_trial{idx + 1:02d}"
                media_path: str = f"{self.base_path}videos/{uid}/{eid}.avi"

                clip = VideoFileClip(media_path)
                fps, size = clip.fps, clip.size
                yield FlexibleDatasetPoint(
                    eid,
                    EEG(data=trial, fs=128, eid=eid).as_mod_tuple(),
                    Video(data=clip, fps=fps, resolution=size, eid=eid).as_mod_tuple(),
                    ("labels", {"values": labels}),
                )
