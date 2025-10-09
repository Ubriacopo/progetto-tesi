import numpy as np

from core_data.media.ecg.ecg import ECG


class AmigosConfig:
    """
    Static information on AMIGOS.
    """

    class Video:
        fps: int = 25

    class EEG:
        fs: int = 128

    class Audio:
        fs = 44100

    EEG_CHANNELS = [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
        "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
    ]

    ECG_CHANNELS = ["ECG Right", "ECG Left"]
    OTHER_CHANNELS = ["GSR"]

    CH_NAMES = EEG_CHANNELS + ECG_CHANNELS + OTHER_CHANNELS

    CH_TYPES = ["eeg"] * len(EEG_CHANNELS) + ["ecg"] * len(ECG_CHANNELS) + ["misc"] * len(OTHER_CHANNELS)
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

        # shape [12, T] to be compliant with ECG-LM requirements
        # todo axis=0 se no time seq
        signal_12xT = np.stack(
            [I, II, III, aVR, aVL, aVF, zeros, zeros, zeros, zeros, zeros, zeros], axis=1
        )

        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        ecg.leads = lead_names
        ecg.data = signal_12xT

        return ecg
