from pathlib import Path
from typing import Iterator

import numpy as np
from moviepy import VideoFileClip

from common.data.audio import Audio
from common.data.data_point import EEGDatasetDataPoint
from common.data.eeg import EEG
from common.data.loader import DataPointsLoader
from common.data.video import Video


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
            for idx, (labels, trial) in enumerate(zip(data["labels"], data["data"])):
                eid: str = f"{uid}_trial{idx + 1:02d}"
                media_path: str = f"{self.base_path}videos/{uid}/{eid}.avi"

                clip = VideoFileClip(media_path)
                # todo fare piu dict like cosi meno rigido?
                yield EEGDatasetDataPoint(
                    entry_id=eid,
                    # Ricorda che noi abbiamo piu channel di quelli che sono realmente di EEG
                    eeg=EEG(data=trial, file_path=None, fs=128, entry_id=eid),
                    vid=Video(data=clip, file_path=media_path, fps=clip.fps, resolution=clip.size, entry_id=eid),
                    aud=Audio(data=clip.audio, file_path=media_path, fs=128, entry_id=eid)  # todo correct fs
                )
