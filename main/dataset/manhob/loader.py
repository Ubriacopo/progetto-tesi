from pathlib import Path
from typing import Optional

import mne
from mne.io.edf.edf import RawEDF
from moviepy import VideoFileClip

from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.loader import DataPointsLoader
from main.core_data.media.audio import Audio
from main.core_data.media.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.dataset.manhob.config import ManhobConfig
from main.dataset.utils import DatasetUidStore


class ManhobPointsLoader(DataPointsLoader):
    DATASET_ID: int = 2

    def __init__(self, base_path: str, dataset_uid_store: DatasetUidStore, config: ManhobConfig = ManhobConfig()):
        super().__init__(dataset_uid_store)
        self.base_path: str = base_path
        self.config: ManhobConfig = config

    def scan(self):
        # In MANHOB we have folders that are experiments.
        processed_data = Path(self.base_path)
        for i in processed_data.iterdir():
            try:
                experiment_id = i.stem  # MANHOB experiment ID

                raw: Optional[RawEDF] = None
                clip: Optional[VideoFileClip] = None

                for file in i.iterdir():
                    if file.suffix == ".bdf":
                        raw = mne.io.read_raw_bdf(str(file), preload=True)
                    elif file.suffix == ".avi":
                        clip = VideoFileClip(str(file))

                # MANHOB always has both so we might match errors
                assert clip is not None and raw is not None, f"Problem was met, the experiment {experiment_id} misses a modality"

                nei = self.dataset_uid_store.uid(experiment_id, experiment_id, "amigos")
                metadata = {"nei": nei, "dataset_id": self.DATASET_ID}
                # Store the current to fs so that we have it ready
                self.dataset_uid_store.store_dictionary()
                eeg_fs = self.config.eeg_source_config.fs
                ecg_leads = self.config.ecg_source_config.LEAD_NAMES
                yield FlexibleDatasetPoint(
                    experiment_id,
                    EEG(eid=experiment_id, data=raw.copy().pick(["eeg"]), fs=eeg_fs).as_mod_tuple(),
                    ECG(eid=experiment_id, data=raw.copy().pick(["ecg"]), fs=eeg_fs, leads=ecg_leads).as_mod_tuple(),
                    Video(data=clip, fps=clip.fps, resolution=clip.size, eid=experiment_id).as_mod_tuple(),
                    # TODO Vedi se esiste audio
                    Audio(data=clip.audio, fs=clip.audio.fps, eid=experiment_id).as_mod_tuple(),
                    Text(eid=experiment_id, data=clip.audio.copy(), base_audio=clip.audio.copy()).as_mod_tuple(),
                    Metadata(data=metadata, eid=experiment_id).as_mod_tuple()
                    # No assessment! TODO Vedi se rompe objective
                )
            except Exception as e:
                # TODO robust logging
                print(f"Loading failed for {i.stem}. Procedure will continue and drop the element")
                print(e)
