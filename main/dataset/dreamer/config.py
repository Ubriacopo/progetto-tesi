import dataclasses

from main.core_data.media.eeg.config import EegSourceConfig
from main.dataset.base_config import DatasetConfig


@dataclasses.dataclass
class DreamerEegSourceConfig(EegSourceConfig):
    fs: int = dataclasses.field(default=128)
    EEG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
    ])

    ECG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [])
    OTHER_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [])


@dataclasses.dataclass
class DreamerConfig(DatasetConfig):
    eeg_source_config: DreamerEegSourceConfig = dataclasses.field(default_factory=DreamerEegSourceConfig)
