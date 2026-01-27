import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import mne
from mne.io import RawArray
from mne.io.edf.edf import RawEDF
from moviepy import VideoFileClip

from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.loader import DataPointsLoader
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.metadata.metadata import Metadata, MetaObject
from main.core_data.media.video import Video
from main.dataset.mahnob.config import MahnobConfig
from main.dataset.utils import DatasetUidStore


class MahnobPointsLoader(DataPointsLoader):
    DATASET_ID: int = 4

    def __init__(self, base_path: str, dataset_uid_store: DatasetUidStore, config: MahnobConfig = MahnobConfig()):
        super().__init__(dataset_uid_store)
        self.base_path: str = base_path
        self.config: MahnobConfig = config
        self.length: int = 0

    def __len__(self) -> int:
        if self.length == 0:
            folder = Path(self.base_path)
            self.length = sum(1 for _ in folder.iterdir())

        return self.length

    def scan(self):
        # In Manhob we have folders that are experiments.
        processed_data = Path(self.base_path)
        for i in processed_data.iterdir():
            try:
                if i.stem == "EEGAVI-processed":
                    continue  # This folder is to ignore.

                experiment_id = i.stem  # Manhob experiment ID

                raw: Optional[RawEDF] = None
                clip: Optional[VideoFileClip] = None

                participant_id: int = None

                offset = 30  # Delay of videocamera start
                for file in i.iterdir():
                    if file.suffix == ".bdf":
                        raw: RawEDF = mne.io.read_raw_bdf(str(file), preload=True)
                        data, info = raw.get_data(), raw.info
                        events = mne.find_events(raw)
                        if len(events) > 0:
                            # First event should always match to the delay of videocamera start.
                            offset = (events[0] / info['sfreq'])[0]
                        raw: RawArray = mne.io.RawArray(data, info)


                    elif file.suffix == ".avi":
                        clip = VideoFileClip(str(file))
                        participant_id = int(file.name.split("-")[0][1:])

                # Manhob always has both so we might match errors
                assert clip is not None and raw is not None, f"Problem was met, the experiment {experiment_id} misses a modality"

                nei = self.dataset_uid_store.uid(experiment_id, experiment_id, "MANHOB")
                metadata = MetaObject(
                    experiment=nei, dataset_id=self.DATASET_ID, person_id=participant_id, trial=experiment_id
                )

                # Store the current to fs so that we have it ready
                self.dataset_uid_store.store_dictionary()
                yield FlexibleDatasetPoint(
                    nei,
                    EEG(eid=nei, data=raw.copy().pick(["eeg"]), fs=raw.info['sfreq']).as_mod_tuple(),
                    ECG(eid=nei, data=raw.copy().pick(self.config.eeg_source_config.ECG_CHANNELS),
                        fs=raw.info['sfreq'], leads=self.config.ecg_source_config.LEAD_NAMES).as_mod_tuple(),
                    # All MANHOB videos have 30s offset
                    Video(data=clip, fps=clip.fps, resolution=clip.size, eid=nei,
                          offset=offset, filepath=clip.filename).as_mod_tuple(),
                    Metadata(data=asdict(metadata), eid=nei).as_mod_tuple()
                )
            except Exception as e:
                self.logger.info(f"Loading failed for {i.stem}. Procedure will continue and drop the element")
                self.logger.error(e)
