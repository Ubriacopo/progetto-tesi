import re
from pathlib import Path
from typing import Iterator

import mne
import numpy as np
import pandas as pd
from moviepy import VideoFileClip

from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.metadata.metadata import Metadata
from main.dataset.amigos.config import AmigosConfig
from main.dataset.amigos.utils import extract_trial_data, load_participant_data
from main.core_data.media.audio.audio import Audio
from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.media.ecg.ecg import ECG
from main.core_data.media.eeg import EEG
from main.core_data.loader import DataPointsLoader
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.dataset.utils import DatasetUidStore


class AmigosPointsLoader(DataPointsLoader):
    def __init__(self, base_path: str, dataset_uid_store: DatasetUidStore):
        super().__init__()
        self.base_path: str = base_path
        self.dataset_uid_store: DatasetUidStore = dataset_uid_store
        self.config = AmigosConfig()

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
            try:
                pat = re.compile(r'^P\(\d+(?:,\d+)*\)_\w\d+_face$')
                if pat.match(v.stem):
                    print("These files are ignored as we have to extract the person face from them. TODO if not done.")
                    continue

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
                eeg_fs = self.config.eeg_source_config.fs
                info = mne.create_info(
                    ch_names=self.config.eeg_source_config.get_CH_NAMES(),
                    ch_types=self.config.eeg_source_config.get_CH_TYPES(),
                    sfreq=eeg_fs
                )

                raw = mne.io.RawArray(eeg_data[0].T, info=info, verbose=False)

                nei = self.dataset_uid_store.uid(person[1:], video_id, "amigos")
                metadata = {"nei": nei, "dataset_id": 0}

                yield FlexibleDatasetPoint(
                    experiment_id,
                    EEG(eid=experiment_id, data=raw.copy().pick(["eeg"]), fs=eeg_fs, ).as_mod_tuple(),
                    ECG(eid=experiment_id,
                        data=raw.copy().pick(["ecg"]), fs=eeg_fs,
                        leads=self.config.ecg_source_config.LEAD_NAMES,
                        patient_gender=next(iter(user_metadata["Gender"].values())).upper(),
                        patient_age=next(iter(user_metadata["Age"].values())), ).as_mod_tuple(),
                    Video(data=clip, fps=clip.fps, resolution=clip.size, eid=experiment_id).as_mod_tuple(),
                    Audio(data=clip.audio, fs=clip.audio.fps, eid=experiment_id).as_mod_tuple(),
                    Text(eid=experiment_id, data=clip.audio.copy(), base_audio=clip.audio.copy()).as_mod_tuple(),
                    Assessment(data=assessments[0][0], eid=experiment_id).as_mod_tuple(),
                    Metadata(data=metadata, eid=experiment_id).as_mod_tuple()
                )

            except Exception as e:
                # TODO robust logging
                print(f"Loading failed for {v.stem}. Procedure will continue and drop the elemnt")
                print(e)

        self.dataset_uid_store.store_dictionary()
