import dataclasses

import numpy as np

from main.core_data.media.audio.config import AudSourceConfig
from main.core_data.media.ecg.config import EcgSourceConfig
from main.core_data.media.ecg.ecg import ECG
from main.core_data.media.eeg.config import EegSourceConfig
from main.core_data.media.text.config import TxtSourceConfig
from main.core_data.media.video.config import VidSourceConfig
from main.dataset.base_config import DatasetConfig


@dataclasses.dataclass
class AmigosEegSourceConfig(EegSourceConfig):
    fs: int = 128
    EEG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
    ])

    ECG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: ["ECG Right", "ECG Left"])
    OTHER_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: ["GSR"])


class AmigosEcgSourceConfig(EcgSourceConfig):
    @staticmethod
    def prepare_ecg(ecg: ECG) -> ECG:
        # RA-LL  -> Lead II
        ecg_II = ecg.data[:, 0, :]
        # LA-LL  -> Lead III
        ecg_III = ecg.data[:, 1, :]

        I = ecg_II - ecg_III
        II = ecg_II
        III = ecg_III
        aVR = -(I + II) / 2
        aVL = I - II / 2
        aVF = II - I / 2
        zeros = np.zeros_like(I)

        # shape [12, T] to be compliant with ECG-LM requirements
        # todo axis=0 se no time seq
        signal_12xT = np.stack([I, II, III, aVR, aVL, aVF, zeros, zeros, zeros, zeros, zeros, zeros], axis=1)
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        ecg.leads = lead_names
        ecg.data = signal_12xT

        return ecg


@dataclasses.dataclass
class AmigosConfig(DatasetConfig):
    eeg_source_config: AmigosEegSourceConfig = dataclasses.field(default_factory=AmigosEegSourceConfig)
    aud_source_config: AudSourceConfig = dataclasses.field(default_factory=lambda: AudSourceConfig(fs=44100))
    vid_source_config: VidSourceConfig = dataclasses.field(default_factory=lambda: VidSourceConfig(fps=25))
    ecg_source_config: AmigosEcgSourceConfig = dataclasses.field(
        default_factory=lambda: AmigosEcgSourceConfig(LEAD_NAMES=["II", "III"])
    )
    txt_source_config: TxtSourceConfig = dataclasses.field(default_factory=TxtSourceConfig)
