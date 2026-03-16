import dataclasses

from main.core_data.media.eeg.config import EegSourceConfig
from main.dataset.base_config import DatasetConfig


@dataclasses.dataclass
class FacedEegSourceConfig(EegSourceConfig):
    fs: int = 250
    EEG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [
        "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6", "Cz", "C3", "C4", "T7", "T8",
        "CP1", "CP2", "CP5", "CP6", "Pz", "P3", "P4", "P7", "P8", "PO3", "PO4", "Oz", "O1", "O2", "HEOR", "HEOL"
    ])

    ECG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [
    ])

    OTHER_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [
    ])


@dataclasses.dataclass
class FacedScoreLabels:
    labels: list[str] = dataclasses.field(default_factory=lambda: [
        # Emotions
        "joy",
        "tenderness",
        "inspiration",
        "amusement",
        "anger",
        "disgust",
        "fear",
        "sadness",
        # V/A
        "arousal",
        "valence",
        "familiarity",
        "liking"
    ])

    rating_scales: list[tuple[int | float, int | float]] = dataclasses.field(default_factory=lambda: [
        # They all are in the [0-7] range
        ((0., 7.),) * 12
    ])


@dataclasses.dataclass
class FacedConfig(DatasetConfig):
    eeg_source_config: FacedEegSourceConfig = dataclasses.field(default_factory=lambda: FacedEegSourceConfig())
    score_labels: FacedScoreLabels = dataclasses.field(default_factory=lambda : FacedScoreLabels())
