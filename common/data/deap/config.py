class DeapConfig:
    class EEG:
        original_fs: int = 128

    class Video:
        fps: int = 50

    CH_NAMES = [
        # EEG Channels (8x4)
        "FP1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
        "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
        "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz",
        "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2",
        # Others
        "hEOG", "vEOG", "zEMG", "tEMG", "GSR", "Respiration belt", "Plethysmograph", "Temperature"
    ]

    CH_TYPES = ["eeg"] * 32 + ["misc"] * 8
