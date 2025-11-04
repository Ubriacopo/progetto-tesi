import dataclasses
from typing import Optional

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

    output_max_length: int
    preprocessing_function: str  # What functions to call to make processing start.
    preprocessing_pipeline: str  # Pipeline to call inside the preprocessing function
    eeg_config: Optional[EegTargetConfig] = dataclasses.field(default_factory=EegTargetConfig)
    ecg_config: Optional[EcgTargetConfig] = dataclasses.field(default_factory=EcgTargetConfig)
    aud_config: Optional[AudTargetConfig] = dataclasses.field(default_factory=AudTargetConfig)
    vid_config: Optional[VidTargetConfig] = dataclasses.field(default_factory=VidTargetConfig)
    txt_config: Optional[TxtTargetConfig] = dataclasses.field(default_factory=TxtTargetConfig)
