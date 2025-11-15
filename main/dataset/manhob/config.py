import dataclasses

from main.core_data.media.ecg import ECG
from main.core_data.media.ecg.config import EcgSourceConfig
from main.core_data.media.eeg.config import EegSourceConfig
from main.core_data.media.video.config import VidSourceConfig
from main.dataset.base_config import DatasetConfig


class ManhobEegSourceConfig(EegSourceConfig):
    fs: int = 256
    EEG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [
        'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2',
        'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2',
    ])

    ECG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [
        'EXG1', 'EXG2', 'EXG3'  # ECG leads (ECG1, ECG2, ECG3)
    ])

    OTHER_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [
        'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Temp'
    ])


class ManhobEcgSourceConfig(EcgSourceConfig):
    LEAD_NAMES: list[str] = dataclasses.field(default_factory=lambda: [])

    @staticmethod
    def prepare_ecg(ecg: ECG) -> ECG:
        pass  # TODO implement


class ManhobConfig(DatasetConfig):
    eeg_source_config: ManhobEegSourceConfig = dataclasses.field(default_factory=ManhobEegSourceConfig)
    vid_source_config: VidSourceConfig = dataclasses.field(default_factory=lambda: VidSourceConfig(fps=61))
    ecg_source_config: ManhobEcgSourceConfig = dataclasses.field(default_factory=ManhobEegSourceConfig)
