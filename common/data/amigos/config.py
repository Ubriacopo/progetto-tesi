class AmigosConfig:
    """
    Static information on AMIGOS.
    """
    CH_NAMES = [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",  # EEG Channels
        "ECG Right", "ECG Left", "GSR"  # Others (ECG + ECG + MISC)
    ]

    CH_TYPES = ["eeg"] * 14 + ["ecg"] * 2 + ["misc"]

    original_aud_fs = 44100
