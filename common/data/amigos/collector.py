import dataclasses
import re
from pathlib import Path

import numpy as np

from common.data.amigos.utils import extract_trial_data, load_participant_data
from common.data.collector import DatasetDataCollection, DataCollector


@dataclasses.dataclass
class AMIGOSDatasetDataCollection(DatasetDataCollection):
    entry_id: str
    eeg_data: list[np.ndarray] | np.ndarray
    mediafile_path: str | Path


class AMIGOSCollector(DataCollector):
    def __init__(self, base_path: str):
        super().__init__()
        self.base_path: str = base_path

    def _scan(self, *args, **kwargs) -> list[DatasetDataCollection]:
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
            self.data.append(AMIGOSDatasetDataCollection(experiment_id, eeg_data[0], str(v.resolve())))

        self.scanned = True
        return self.data
