import dataclasses

import numpy as np

from main.core_data.media.assessment.assessment import AssessmentLabels
from main.core_data.media.assessment.config import ScoreLabelsConfig
from main.core_data.media.ecg import ECG
from main.core_data.media.ecg.config import EcgSourceConfig
from main.core_data.media.eeg.config import EegSourceConfig
from main.dataset.base_config import DatasetConfig


@dataclasses.dataclass
class ClareEegSourceConfig(EegSourceConfig):
    fs: int = dataclasses.field(default=256)
    # AF7, AF8, TP9 and TP10
    EEG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: ["Fp1", "Fp2", "T7", "T8"])

    ECG_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [])
    OTHER_CHANNELS: list[str] = dataclasses.field(default_factory=lambda: [])

    REMAP: dict[str, str] = dataclasses.field(default_factory=lambda: {
        "AF7": "Fp1", "AF8": "Fp2", "TP9": "T7", "TP10": "T8",
    })

    def channels_remap(self, values: dict[str, float]) -> dict[str, float]:
        return_channels: dict = {}
        for channel, value in values.items():
            return_channels[self.REMAP[channel]] = value
        return return_channels


@dataclasses.dataclass
class ClareEcgSourceConfig(EcgSourceConfig):
    LEAD_NAMES: list[str] = dataclasses.field(default_factory=lambda: ["RA", "LA", "LL"])

    @staticmethod
    def prepare_ecg(ecg: ECG) -> ECG:
        """
        Build an ECG-LM-compatible 12-lead tensor from available leads.

        Expected input channels in ecg.data:
            channel 0 -> ECG LL-RA CAL  == Lead II
            channel 1 -> ECG LA-RA CAL  == Lead I
            channel 2 -> ECG Vx-RL CAL  == one unknown precordial lead candidate

        Input shape:
            [B, C, T]

        Output shape:
            [B, 12, T]
        """
        import numpy as np

        if ecg.data.ndim != 3:
            raise ValueError(f"Expected ecg.data with shape [B, C, T], got {ecg.data.shape}")

        if ecg.data.shape[1] < 2:
            raise ValueError(
                "Need at least 2 channels: LL-RA (Lead II) and LA-RA (Lead I)"
            )

        # Available limb leads
        II = ecg.data[:, 0, :]  # LL-RA
        I = ecg.data[:, 1, :]  # LA-RA

        # Derived limb leads
        III = II - I
        aVR = -(I + II) / 2.0
        aVL = I - II / 2.0
        aVF = II - I / 2.0

        zeros = np.zeros_like(I)

        # Precordial leads: unknown unless Vx identity is known
        V1 = zeros
        V2 = zeros
        V3 = zeros
        V4 = zeros
        V5 = zeros
        V6 = zeros

        ecg.leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        ecg.data = np.stack([I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6], axis=1)
        return ecg


@dataclasses.dataclass
class ClareScoreLabelsConfig(ScoreLabelsConfig):
    labels: list[str] = dataclasses.field(default_factory=lambda: ["workload", ])

    scales: list[tuple[int | float, int | float]] = dataclasses.field(
        default_factory=lambda: list(((1, 9),) * 1)  # (1-9) + (0-9)
    )


@dataclasses.dataclass
class ClareConfig(DatasetConfig):
    eeg_source_config: ClareEegSourceConfig = dataclasses.field(default_factory=ClareEegSourceConfig)
    ecg_source_config: ClareEcgSourceConfig = dataclasses.field(default_factory=ClareEcgSourceConfig)
    score_labels_config: ClareScoreLabelsConfig = dataclasses.field(default_factory=lambda: ClareScoreLabelsConfig())
