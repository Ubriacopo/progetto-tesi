import dataclasses
from pathlib import Path
from typing import Optional, Any

import pandas as pd

from main.core_data.media.audio import AudTargetConfig
from main.core_data.media.ecg import EcgTargetConfig
from main.core_data.media.eeg import EegTargetConfig
from main.core_data.media.text import TxtTargetConfig
from main.core_data.media.video import VidTargetConfig


@dataclasses.dataclass
class IntervalsExtractorConfig:
    segmenter_args: dict[str, Any]
    segmenter_type: str


@dataclasses.dataclass
class PreprocessingConfig:
    extraction_data_folder: str  # Subpath to where extracted intervals are placed
    output_path: str  # Subpath to where output has to go to.
    out_folder_name: str

    output_max_length: int
    preprocessing_pipeline: str  # Pipeline to call inside the preprocessing function


@dataclasses.dataclass
class DatasetConfig:
    name: str

    config_classpath: str
    loader_classpath: str

    base_path: str  # Where things are fetched from
    data_path: str  # Subpath to where the dataset is placed
    uid_store_path: str

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
            self.df = pd.read_csv(file_path, index_col=None, dtype={"user_id": "string"})

        self.next_id = (self.df["id"].max() + 1) if len(self.df) else 0

    def uid(self, user_id: str, experiment_id: str, dataset_name: str) -> int:
        next_id = self.next_id

        exists = self.df[(self.df["user_id"] == user_id)
                         & (self.df["experiment_id"] == experiment_id)
                         & (self.df["dataset_name"] == dataset_name)]
        if len(exists) > 0:
            return exists.iloc[0]['id']

        self.df.loc[len(self.df)] = [next_id, user_id, experiment_id, dataset_name]
        self.next_id = (self.df["id"].max() + 1) if len(self.df) else 0
        return next_id

    def restore_id(self, eid: int) -> dict:
        row = self.df.loc[self.df["id"] == eid]
        return row.iloc[0].to_dict() if len(row) else None

    def store_dictionary(self):
        self.df.to_csv(self.path, index=False)
