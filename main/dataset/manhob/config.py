import dataclasses

import numpy as np

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
    # todo cambia nome non sono lead names ma channels
    LEAD_NAMES: list[str] = dataclasses.field(default_factory=lambda: ["RA", "LA", "LL"])

    @staticmethod
    def prepare_ecg(ecg: ECG) -> ECG:
        RA = ecg.data[:, 0, :]
        LA = ecg.data[:, 1, :]
        LL = ecg.data[:, 2, :]

        II = LL - RA
        III = LL - LA
        I = LA - RA

        aVR = -(I + II) / 2
        aVL = I - (II / 2)
        aVF = II - (I / 2)
        zeros = np.zeros_like(I)

        # shape [12, T] to be compliant with ECG-LM requirements
        signal_12xT = np.stack([I, II, III, aVR, aVL, aVF, zeros, zeros, zeros, zeros, zeros, zeros], axis=1)
        LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

        ecg.leads = LEADS
        ecg.data = signal_12xT

        return ecg


class ManhobConfig(DatasetConfig):
    eeg_source_config: ManhobEegSourceConfig = dataclasses.field(default_factory=ManhobEegSourceConfig)
    vid_source_config: VidSourceConfig = dataclasses.field(default_factory=lambda: VidSourceConfig(fps=61))
    ecg_source_config: ManhobEcgSourceConfig = dataclasses.field(default_factory=ManhobEegSourceConfig)
