import re
from pathlib import Path
from typing import Iterator

import mne
import numpy as np
import pandas as pd
from moviepy import VideoFileClip

from main.core_data.media.assessment.assessment import Assessment
from main.dataset.amigos.config import AmigosConfig
from main.dataset.amigos.utils import extract_trial_data, load_participant_data
from main.core_data.media.audio.audio import Audio
from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.media.ecg.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.loader import DataPointsLoader
from main.core_data.media.text import Text
from main.core_data.media.video import Video


class AmigosPointsLoader(DataPointsLoader):
    def __init__(self, base_path: str):
        super().__init__()
        self.base_path: str = base_path

    def scan(self) -> Iterator[FlexibleDatasetPoint]:
        processed_data = Path(self.base_path + "pre_processed_py/")

        if not processed_data.exists():
            for f in Path(self.base_path + "pre_processed").iterdir():
                extract_trial_data(self.base_path + "pre_processed_py/", str(f))

        participant_data = load_participant_data(Path(self.base_path + "pre_processed_py/"))
        participant_metadata = pd.read_excel(self.base_path + "Metadata_xlsx/Participant_Questionnaires.xlsx")

        face_video_folder = self.base_path + "face/"
        face_folder = Path(face_video_folder)

        for v in face_folder.iterdir():
            # [0] -> P40 [1] -> 18 [2] -> face(.mov) (Stemmed)
            person, video_id, _ = v.stem.split("_")
            user_metadata = participant_metadata[participant_metadata["UserID"] == int(person[1:])].to_dict()
            # Add missing prefix zero to match the np data
            person = re.sub(r'([A-Z])(\d)\b', r'\g<1>0\2', person)

            experiment_id = person + "_" + video_id

            video_index = np.where(participant_data[person]["VideoIDs"] == video_id)[0]
            eeg_data = participant_data[person]["joined_data"][video_index]
            assessments = participant_data[person]["labels_selfassessment"][video_index]

            media_path: str = str(v.resolve())
            clip = VideoFileClip(media_path)

            # Extract ECG and EEG data + metadata that could be useful
            fs = AmigosConfig.EEG.fs
            info = mne.create_info(ch_names=AmigosConfig.CH_NAMES, ch_types=AmigosConfig.CH_TYPES, sfreq=fs)
            raw = mne.io.RawArray(eeg_data[0].T, info=info, verbose=False)

            # Take from Audio
            yield FlexibleDatasetPoint(
                experiment_id,
                EEG(eid=experiment_id, data=raw.copy().pick(["eeg"]), fs=AmigosConfig.EEG.fs, ).as_mod_tuple(),
                ECG(eid=experiment_id, data=raw.copy().pick(["ecg"]), fs=AmigosConfig.EEG.fs,
                    leads=AmigosConfig.LEAD_NAMES, patient_gender=user_metadata["Gender"][0].upper(),
                    patient_age=user_metadata["Age"][0], ).as_mod_tuple(),
                Video(data=clip, fps=clip.fps, resolution=clip.size, eid=experiment_id).as_mod_tuple(),
                Audio(data=clip.audio, fs=clip.audio.fps, eid=experiment_id).as_mod_tuple(),
                Text(eid=experiment_id, data=clip.audio.copy(), base_audio=clip.audio.copy()).as_mod_tuple(),
                Assessment(data=assessments[0][0], eid=experiment_id).as_mod_tuple(),
            )
