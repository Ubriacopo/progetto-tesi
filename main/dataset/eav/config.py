import dataclasses

from main.core_data.media.assessment.config import ScoreLabelsConfig, CategoricalLabelsConfig
from main.core_data.media.audio.config import AudSourceConfig
from main.core_data.media.eeg.config import EegSourceConfig
from main.core_data.media.video.config import VidSourceConfig
from main.dataset.base_config import DatasetConfig


@dataclasses.dataclass
class EavEegSourceConfig(EegSourceConfig):
    fs: int = 500
    EEG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5',
        'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
    ])

    ECG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [])

    OTHER_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [])


@dataclasses.dataclass
class EavCategorialLabelsConfig(CategoricalLabelsConfig):
    labels: list[str] = dataclasses.field(default_factory=lambda: [
        # L = Listening and S = Speaking
        "Neutral-L", "Neutral-S",
        "Sad-L", "Sad-S",
        "Anxious-L", "Anxious-S",
        "Happy-L", "Happy-S",
        "Calm-L", "Calm-S"
    ])


@dataclasses.dataclass
class EavConfig(DatasetConfig):
    eeg_source_config: EavEegSourceConfig = dataclasses.field(default_factory=EavEegSourceConfig)
    aud_source_config: AudSourceConfig = dataclasses.field(default_factory=lambda: AudSourceConfig(fs=44100))
    vid_source_config: VidSourceConfig = dataclasses.field(default_factory=lambda: VidSourceConfig(fps=30))
    labels_config: EavCategorialLabelsConfig = dataclasses.field(default_factory=lambda: EavCategorialLabelsConfig())