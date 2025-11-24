import logging
import re
from pathlib import Path
from typing import Optional

import mne
from mne.io import RawArray
from mne.io.edf.edf import RawEDF
from moviepy import VideoFileClip, AudioFileClip
from scipy.io import loadmat
from sympy.physics.mechanics.functions import inertia_of_point_mass

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
            # Each folder is a subject
            subject_id = i.stem.split('subject')[1]
            matlab_file = loadmat(f"{i}/EEG/subject{subject_id}_eeg.mat")

            eeg = matlab_file["seg"]
            # TODO: Use them or not? Prolly not.
            labels_mat = loadmat(f"{i}/EEG/subject{subject_id}_eeg_label.mat")

            try:
                # Video files for this dataset are the double of audio files
                for video_file in Path(str(i) + "/Video").iterdir():
                    information = video_file.stem.split('_')
                    index = int(information[0])  # Incremental index that means basically nothing.
                    trial_id = information[2]  # What trial it is (ID)
                    speaking = information[3].lower() == 'speaking'  # If it is speaking means we have audio.
                    emotion = information[4]  # Category of the emotion that should be observed

                    raw: RawArray
                    clip: VideoFileClip = VideoFileClip(str(video_file))
                    audio: Optional[AudioFileClip] = None
                    # TODO hanno fatto puttanate. index é unica cosa da guardare.
                    if speaking:
                        pattern = re.compile(r'^data_\d+\.csv$')
                        audio_path = Path(str(i) + "/Audio")
                        matches = [f for f in audio_path.iterdir() if f.is_file() and pattern.match(f.name)]
                        if len(matches) > 1:
                            raise ValueError("I found more than one audio file for a single clip.")
                        if len(matches) == 0:
                            raise ValueError("I found no audio file while I expected one")

                        audio: Optional[AudioFileClip] = AudioFileClip(str(i) + "/Audio/" + matches[0].stem)

                    nei = self.dataset_uid_store.uid(subject_id, str(trial_id) + "_" + emotion, "EAV")
                    metadata = {"nei": nei, "dataset_id": self.DATASET_ID}

                    # EEG data part
                    info = mne.create_info(
                        ch_names=self.config.eeg_source_config.get_CH_NAMES(),
                        ch_types=self.config.eeg_source_config.get_CH_TYPES(),
                        sfreq=self.config.eeg_source_config.fs
                    )

                    raw = mne.io.RawArray(data, info=info, verbose=False)

                    yield FlexibleDatasetPoint(
                        nei,
                        EEG(eid=nei, data=raw.copy().pick(["eeg"]), fs=raw.info['sfreq']).as_mod_tuple(),
                        Video(eid=nei, ).as_mod_tuple(),
                        Metadata(data=metadata, eid=str(nei)).as_mod_tuple()
                    )

                experiment_id = i.stem  # Manhob experiment ID

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
