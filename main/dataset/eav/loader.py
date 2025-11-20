import logging
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
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.video import Video
from main.dataset.eav.config import EavConfig
from main.dataset.utils import DatasetUidStore


class EavPointsLoader(DataPointsLoader):
    DATASET_ID: int = 2

    def __init__(self, base_path: str, dataset_uid_store: DatasetUidStore, config: EavConfig = EavConfig()):
        super().__init__(dataset_uid_store)
        self.base_path: str = base_path
        self.config: EavConfig = config

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

                for file in i.iterdir():
                    if file.suffix == ".bdf":
                        raw: RawEDF = mne.io.read_raw_bdf(str(file), preload=True)
                        # TODO mne.find_events(raw) mi trova eventi coerenti.
                        #       Problema non posso essere certo su quale dei due sia start di video.
                        #       Dovrei farmi dare resources
                        data, info = raw.get_data(), raw.info
                        raw: RawArray = mne.io.RawArray(data, info)

                    elif file.suffix == ".avi":
                        clip = VideoFileClip(str(file))

                # Manhob always has both so we might match errors
                assert clip is not None and raw is not None, f"Problem was met, the experiment {experiment_id} misses a modality"

                nei = self.dataset_uid_store.uid(experiment_id, experiment_id, "amigos")
                metadata = {"nei": nei, "dataset_id": self.DATASET_ID}
                # Store the current to fs so that we have it ready
                self.dataset_uid_store.store_dictionary()
                yield FlexibleDatasetPoint(
                    experiment_id,
                    EEG(eid=experiment_id, data=raw.copy().pick(["eeg"]), fs=raw.info['sfreq']).as_mod_tuple(),
                    ECG(eid=experiment_id, data=raw.copy().pick(self.config.eeg_source_config.ECG_CHANNELS),
                        fs=raw.info['sfreq'], leads=self.config.ecg_source_config.LEAD_NAMES).as_mod_tuple(),
                    Video(data=clip, fps=clip.fps, resolution=clip.size, eid=experiment_id).as_mod_tuple(),
                    Metadata(data=metadata, eid=experiment_id).as_mod_tuple()
                    # No assessment! TODO Vedi se rompe objective
                )
            except Exception as e:
                logging.info(f"Loading failed for {i.stem}. Procedure will continue and drop the element")
                logging.error(e)
