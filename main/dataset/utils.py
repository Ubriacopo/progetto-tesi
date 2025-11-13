import dataclasses
from pathlib import Path
from typing import Optional

import pandas as pd

from main.core_data.media.audio import AudTargetConfig
from main.core_data.media.ecg import EcgTargetConfig
from main.core_data.media.eeg import EegTargetConfig
from main.core_data.media.text import TxtTargetConfig
from main.core_data.media.video import VidTargetConfig


@dataclasses.dataclass
class PreprocessingConfig:
    dataset_name: str
    base_path: str  # Where things are fetched from
    data_path: str  # Subpath to where the dataset is placed
    extraction_data_folder: str  # Subpath to where extracted intervals are placed
    output_path: str  # Subpath to where output has to go to.
    uid_store_path: str

    output_max_length: int
    preprocessing_function: str  # What functions to call to make processing start.
    preprocessing_pipeline: str  # Pipeline to call inside the preprocessing function
    eeg_config: Optional[EegTargetConfig] = dataclasses.field(default_factory=EegTargetConfig)
    ecg_config: Optional[EcgTargetConfig] = dataclasses.field(default_factory=EcgTargetConfig)
    aud_config: Optional[AudTargetConfig] = dataclasses.field(default_factory=AudTargetConfig)
    vid_config: Optional[VidTargetConfig] = dataclasses.field(default_factory=VidTargetConfig)
    txt_config: Optional[TxtTargetConfig] = dataclasses.field(default_factory=TxtTargetConfig)


class DatasetUidStore:
    def __init__(self, file_path: str):
        self.df = pd.DataFrame(columns=["id", "user_id", "experiment_id", "dataset_name"])
        # Where to store the data
        self.path = file_path

        if Path(file_path).exists():
            self.df = pd.read_csv(file_path, index_col=None)

        self.next_id = len(self.df)

    def uid(self, person_id: str, experiment_id: str, dataset_name: str) -> int:
        next_id = self.next_id

        exists = self.df[(self.df["user_id"] == person_id) & (self.df["experiment_id"] == experiment_id)]
        if len(exists) > 0:
            return exists.iloc[0]['id']

        self.df.loc[len(self.df)] = [next_id, person_id, experiment_id, dataset_name]
        self.next_id += 1
        return next_id

    def restore_id(self, uid: int) -> dict:
        return self.df[uid].to_dict()

    def store_dictionary(self):
        self.df.to_csv(self.path, index=False)
