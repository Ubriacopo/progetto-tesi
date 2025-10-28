from __future__ import annotations

import dataclasses

from omegaconf import MISSING


@dataclasses.dataclass
class EegTargetConfig:
    model_weights_path: str = MISSING
    fs: int = 200  # CBraMod


@dataclasses.dataclass
class EegSourceConfig:
    fs: int

    EEG_CHANNELS: list[str]
    ECG_CHANNELS: list[str]
    OTHER_CHANNELS: list[str]

    def get_CH_NAMES(self):
        return self.EEG_CHANNELS + self.ECG_CHANNELS + self.OTHER_CHANNELS

    def get_CH_TYPES(self):
        return ["eeg"] * len(self.EEG_CHANNELS) + ["ecg"] * len(self.ECG_CHANNELS) + ["misc"] * len(self.OTHER_CHANNELS)
