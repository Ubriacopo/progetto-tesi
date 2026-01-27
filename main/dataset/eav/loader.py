import re
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import mne
from einops import rearrange
from mne.io import RawArray
from moviepy import VideoFileClip, AudioFileClip
from scipy.io import loadmat

from core_data.media.metadata.metadata import MetaObject
from main.core_data.data_point import FlexibleDatasetPoint
from main.core_data.loader import DataPointsLoader
from main.core_data.media.audio import Audio
from main.core_data.media.eeg import EEG
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.dataset.eav.config import EavConfig
from main.dataset.utils import DatasetUidStore


class EavPointsLoader(DataPointsLoader):
    DATASET_ID: int = 2

    def __init__(self, base_path: str, dataset_uid_store: DatasetUidStore, config: EavConfig = EavConfig()):
        super().__init__(dataset_uid_store)
        self.base_path: str = base_path
        self.config: EavConfig = config

        self.data_folder: str = self.base_path + "EAV/"
        self.length: int = 0

    def __len__(self) -> int:
        if self.length == 0:
            folder = Path(self.data_folder)
            for p in folder.iterdir():
                if "subject" in p.stem:
                    self.length += sum(1 for _ in Path(str(p) + "/Video").iterdir())

        return self.length

    def scan(self):
        # In Manhob we have folders that are experiments.
        processed_data = Path(self.base_path + "EAV/")
        for i in processed_data.iterdir():
            try:
                if not "subject" in i.stem:
                    continue
                # Each folder is a subject
                subject_id = i.stem.split('subject')[1]
                matlab_file = loadmat(f"{i}/EEG/subject{subject_id}_eeg.mat")

                if "seg" in matlab_file:
                    eeg = matlab_file["seg"]
                elif "seg1" in matlab_file:
                    # Questo è un vero pasticcio. Qualcuno qui ha fatto pasticci suppongo.
                    eeg = matlab_file["seg1"]
                else:
                    raise ValueError("The matlab file is missing the correct column to elaborate the data."
                                     f"File has: {matlab_file.keys()} and not 'seg'")

                eeg = rearrange(eeg, "d c b -> b c d")
                # Video files for this dataset are the double of audio files
                for video_file in Path(str(i) + "/Video").iterdir():
                    try:
                        clip: VideoFileClip = VideoFileClip(str(video_file))

                        information = video_file.stem.split('_')
                        index = int(information[0])  # Incremental index that identifies the media
                        speaking = information[3].lower() == 'speaking'  # If it is speaking means we have audio.
                        emotion = information[4]  # Category of the emotion that should be observed. No use.

                        audio: Optional[AudioFileClip] = None
                        audio_filepath: Optional[str] = None
                        if speaking:
                            pattern = re.compile(fr'{information[0]}_.*')
                            audio_path = Path(str(i) + "/Audio")
                            matches = [f for f in audio_path.iterdir() if f.is_file() and pattern.match(f.name)]
                            if len(matches) > 1:
                                raise ValueError("I found more than one audio file for a single clip.")
                            if len(matches) == 0:
                                raise ValueError("I found no audio file while I expected one")

                            audio_filepath = str(matches[0])
                            audio: Optional[AudioFileClip] = AudioFileClip(audio_filepath)

                        nei = self.dataset_uid_store.uid(subject_id, str(index) + "_" + emotion, "EAV")
                        # Store the current to fs so that we have it ready
                        self.dataset_uid_store.store_dictionary()
                        metadata = MetaObject(
                            experiment=index, dataset_id=self.DATASET_ID, person_id=subject_id
                        )

                        # EEG data part
                        info = mne.create_info(
                            ch_names=self.config.eeg_source_config.get_CH_NAMES(),
                            ch_types=self.config.eeg_source_config.get_CH_TYPES(),
                            sfreq=self.config.eeg_source_config.fs
                        )

                        raw: RawArray = mne.io.RawArray(eeg[index], info=info, verbose=False)

                        audio_fs = audio.fps if audio is not None else 0
                        audio_copy = audio.copy() if audio is not None else None

                        yield FlexibleDatasetPoint(
                            nei,
                            EEG(eid=nei, data=raw.copy().pick(["eeg"]), fs=raw.info['sfreq']).as_mod_tuple(),
                            Video(eid=nei, data=clip, fps=clip.fps, resolution=clip.size,
                                  filepath=str(video_file.resolve())).as_mod_tuple(),
                            Audio(eid=nei, data=audio, fs=audio_fs, filepath=audio_filepath).as_mod_tuple(),
                            Text(eid=nei, data=audio_copy, base_audio=audio_copy).as_mod_tuple(),
                            Metadata(data=asdict(metadata), eid=nei).as_mod_tuple()
                        )

                    except Exception as e:
                        self.logger.info(
                            f"Loading failed for {i.stem} entry {video_file}. Procedure will continue and drop the element"
                        )
                        self.logger.error(e)

            except Exception as e:
                self.logger.info(f"Loading failed for {i.stem}. Procedure will continue and drop the element")
                self.logger.error(e)
