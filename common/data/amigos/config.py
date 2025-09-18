import numpy as np

from common.data.ecg.ecg import ECG


class AmigosConfig:
    """
    Static information on AMIGOS.
    """
    original_aud_fs = 44100
    original_eeg_fs = 128
    original_vid_fps = 25

    CH_NAMES = [
        # EEG Channels (14)
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
        "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
        # Others (ECG + ECG + MISC)
        "ECG Right", "ECG Left", "GSR"
    ]

    CH_TYPES = ["eeg"] * 14 + ["ecg"] * 2 + ["misc"]
    LEAD_NAMES = ["II", "III"]

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
        # TODO Controlla che sia ffettivamente corretto
        # shape [12, T] to be compliant with ECG-LM requirements
        # todo axis=0 se no time seq
        signal_12xT = np.stack(
            [I, II, III, aVR, aVL, aVF, zeros, zeros, zeros, zeros, zeros, zeros], axis=1
        )

        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        ecg.leads = lead_names
        ecg.data = signal_12xT

        return ecg
