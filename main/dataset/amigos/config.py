import dataclasses

import numpy as np

from main.core_data.media.assessment.assessment import AssessmentLabels
from main.core_data.media.assessment.config import ScoreLabelsConfig
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


@dataclasses.dataclass
class AmigosEcgSourceConfig(EcgSourceConfig):
    @staticmethod
    def prepare_ecg(ecg: ECG) -> ECG:
        # AMIGOS uses Shimmer ECG. Likely channel assignment:
        ecg_right = ecg.data[:, 0, :]
        ecg_left = ecg.data[:, 1, :]

        # ch0 = LL - RA = Lead II
        # ch1 = LA - RA = Lead I
        II = -ecg_right          # LL - RA
        III = -ecg_left          # LL - LA
        I = ecg_left - ecg_right # LA - RA

        aVR = -(I + II) / 2
        aVL = I - II / 2
        aVF = II - I / 2

        zeros = np.zeros_like(I)

        ecg.data = np.stack([I, II, III, aVR, aVL, aVF, zeros, zeros, zeros, zeros, zeros, zeros], axis=1, )
        ecg.leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", ]
        return ecg


@dataclasses.dataclass
class AmigosScoreLabels(ScoreLabelsConfig):
    labels: list[str] = dataclasses.field(default_factory=lambda: [
        # (1-9)
        AssessmentLabels.AROUSAL,
        AssessmentLabels.VALENCE,
        AssessmentLabels.DOMINANCE,
        AssessmentLabels.LIKING,
        AssessmentLabels.FAMILIARITY,
        # (0/1)
        AssessmentLabels.NEUTRAL,
        AssessmentLabels.DISGUST,
        AssessmentLabels.HAPPINESS,
        AssessmentLabels.SURPRISE,
        AssessmentLabels.ANGER,
        AssessmentLabels.FEAR,
        AssessmentLabels.SADNESS
    ])

    scales: list[tuple[int | float, int | float]] = dataclasses.field(
        default_factory=lambda: list(((1., 9.),) * 5 + ((0, 1),) * 7)  # (1-9) + (0-9)
    )


@dataclasses.dataclass
class AmigosConfig(DatasetConfig):
    eeg_source_config: AmigosEegSourceConfig = dataclasses.field(default_factory=AmigosEegSourceConfig)
    aud_source_config: AudSourceConfig = dataclasses.field(default_factory=lambda: AudSourceConfig(fs=44100))
    vid_source_config: VidSourceConfig = dataclasses.field(default_factory=lambda: VidSourceConfig(fps=25))
    ecg_source_config: AmigosEcgSourceConfig = dataclasses.field(
        # TODO verifica perche passo lead names qui se poi prepare fa tutto. Ah sono i nomi dei canali che ho
        default_factory=lambda: AmigosEcgSourceConfig(LEAD_NAMES=["II", "III"])
    )
    txt_source_config: TxtSourceConfig = dataclasses.field(default_factory=TxtSourceConfig)
    score_labels_config: AmigosScoreLabels = dataclasses.field(default_factory=lambda: AmigosScoreLabels())
