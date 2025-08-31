import re
from pathlib import Path
from typing import Iterator

import numpy as np
from moviepy import VideoFileClip

from common.data.amigos.utils import extract_trial_data, load_participant_data
from common.data.audio.audio import Audio
from common.data.eeg import EEG
from common.data.loader import DataLoader
from common.data.data_point import EEGDatasetDataPoint
from common.data.video.video import Video


class AMIGOSLoader(DataLoader):
    def __init__(self, base_path: str):
        super().__init__()
        self.base_path: str = base_path

    def scan(self) -> Iterator[EEGDatasetDataPoint]:
        processed_data = Path(self.base_path + "pre_processed_py/")

        if not processed_data.exists():
            for f in Path(self.base_path + "pre_processed").iterdir():
                extract_trial_data(self.base_path + "pre_processed_py/", str(f))

        participant_data = load_participant_data(Path(self.base_path + "pre_processed_py/"))
        face_video_folder = self.base_path + "face/"
        face_folder = Path(face_video_folder)

        for v in face_folder.iterdir():
            # [0] -> P40 [1] -> 18 [2] -> face(.mov) (Stemmed)
            person, video_id, _ = v.stem.split("_")
            # Add missing prefix zero to match the np data
            person = re.sub(r'([A-Z])(\d)\b', r'\g<1>0\2', person)

            experiment_id = person + "_" + video_id

            video_index = np.where(participant_data[person]["VideoIDs"] == video_id)[0]
            eeg_data = participant_data[person]["joined_data"][video_index]

            media_path: str = str(v.resolve())
            clip = VideoFileClip(media_path)
            vid = Video(data=clip, file_path=media_path, fps=clip.fps, resolution=clip.size, entry_id=experiment_id)
            aud = Audio(data=clip.audio, file_path=media_path, fs=clip.audio.fps, entry_id=experiment_id)
            eeg = EEG(data=eeg_data[0], file_path=None, fs=128, entry_id=experiment_id)

            yield EEGDatasetDataPoint(entry_id=experiment_id, eeg=eeg, vid=vid, aud=aud)
