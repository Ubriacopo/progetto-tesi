from pathlib import Path
from typing import Iterator

import mne
import numpy as np
from moviepy import VideoFileClip

from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.eeg import EEG
from main.core_data.loader import DataPointsLoader
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.video import Video
from main.dataset.deap.config import DeapConfig
from main.dataset.utils import DatasetUidStore


class DeapPointsLoader(DataPointsLoader):
    def __init__(self, base_path: str, dataset_uid_store: DatasetUidStore, config: DeapConfig = DeapConfig()):
        super().__init__(dataset_uid_store)
        self.base_path = base_path
        self.config: DeapConfig = config

        self.length: int = 0

    def __len__(self) -> int:
        if self.length == 0:
            folder = Path(self.base_path + "data_preprocessed_python/")
            self.length = sum(1 for _ in folder.iterdir()) * 40  # 40 Videos for each EEG file

        return self.length

    def scan(self) -> Iterator[FlexibleDatasetPoint]:
        processed_data = Path(self.base_path + "data_preprocessed_python/")
        for i in processed_data.iterdir():
            try:
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

                    nei = self.dataset_uid_store.uid(uid, eid, "deap")
                    # Store the current to fs so that we have it ready
                    self.dataset_uid_store.store_dictionary()

                    # Create EEG data
                    info = mne.create_info(
                        ch_names=self.config.eeg_source_config.get_CH_NAMES(),
                        ch_types=self.config.eeg_source_config.get_CH_TYPES(),
                        sfreq=self.config.eeg_source_config.fs
                    )
                    raw = mne.io.RawArray(trial, info=info, verbose=False)
                    eeg = EEG(eid=nei, data=raw.copy().pick(["eeg"]), fs=raw.info['sfreq'])

                    clip = VideoFileClip(media_path)
                    video = Video(eid=nei, data=clip, fps=clip.fps, resolution=clip.size, filepath=media_path)

                    assessment = Assessment(eid=nei, data=labels)
                    metadata = Metadata(eid=nei, data={"nei": nei, "dataset_id": 3})

                    yield FlexibleDatasetPoint(
                        eid,
                        eeg.as_mod_tuple(),
                        video.as_mod_tuple(),
                        assessment.as_mod_tuple(),
                        metadata.as_mod_tuple()
                    )

            except Exception as e:
                # TODO robust logging
                self.logger.error(f"Loading failed for {i.stem}. Procedure will continue and drop the element")
                self.logger.error(e)
