class AmigosConfig:
    """
    Static information on AMIGOS.
    """
    original_aud_fs = 44100
    original_eeg_fs = 128
    original_vid_fps = 25

    CH_NAMES = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",  # EEG Channels
                "ECG Right", "ECG Left", "GSR"]  # Others (ECG + ECG + MISC)
    CH_TYPES = ["eeg"] * 14 + ["ecg"] * 2 + ["misc"]
