import dataclasses

from main.core_data.media.eeg.config import EegSourceConfig
from main.core_data.media.video.config import VidSourceConfig
from main.dataset.base_config import DatasetConfig


class DeapEegSourceConfig(EegSourceConfig):
    fs: int = 128
    EEG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [
        "FP1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
        "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
    ])

    ECG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [])
    OTHER_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [
        "hEOG", "vEOG", "zEMG", "tEMG", "GSR", "Respiration belt", "Plethysmograph", "Temperature"
    ])


class DeapConfig(DatasetConfig):
    eeg_source_configL: DeapEegSourceConfig = dataclasses.field(default_factory=DeapEegSourceConfig)
    # aud_source_config: AudSourceConfig = dataclasses.field(default_factory=lambda: AudSourceConfig(fs=44100)) Deap has no audio
    vid_source_config: VidSourceConfig = dataclasses.field(default_factory=lambda: VidSourceConfig(fps=24))
