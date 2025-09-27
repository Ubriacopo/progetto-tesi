import re
from pathlib import Path
from typing import Iterator

import mne
import numpy as np
import pandas as pd
from moviepy import VideoFileClip

from common.data.amigos.config import AmigosConfig
from common.data.amigos.utils import extract_trial_data, load_participant_data
from common.data.audio.audio import Audio
from common.data.data_point import FlexibleDatasetPoint
from common.data.ecg.ecg import ECG
from common.data.eeg import EEG
from common.data.loader import DataPointsLoader
from common.data.text import Text
from common.data.video.video import Video


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

            media_path: str = str(v.resolve())
            clip = VideoFileClip(media_path)
            vid = Video(data=clip, fps=clip.fps, resolution=clip.size, eid=experiment_id)
            aud = Audio(data=clip.audio, fs=clip.audio.fps, eid=experiment_id)

            # Extract ECG and EEG data + metadata that could be useful
            info = mne.create_info(
                ch_names=AmigosConfig.CH_NAMES,
                ch_types=AmigosConfig.CH_TYPES,
                sfreq=AmigosConfig.EEG.fs
            )
            raw = mne.io.RawArray(eeg_data[0].T, info=info, verbose=False)

            eeg = EEG(
                eid=experiment_id,
                data=raw.copy().pick(["eeg"]),
                fs=AmigosConfig.EEG.fs,
            )

            ecg = ECG(
                # Signal Data
                eid=experiment_id,
                data=raw.copy().pick(["ecg"]),
                fs=AmigosConfig.EEG.fs,
                # ECG Specific
                leads=AmigosConfig.LEAD_NAMES,
                patient_gender=user_metadata["Gender"][0].upper(),
                patient_age=user_metadata["Age"][0],
            )

            # Take from Audio
            yield FlexibleDatasetPoint(
                experiment_id,
                eeg.as_mod_tuple(),
                ecg.as_mod_tuple(),
                vid.as_mod_tuple(),
                aud.as_mod_tuple(),
            )
