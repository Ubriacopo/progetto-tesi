import dataclasses
from typing import Optional

from main.core_data.media.audio import AudTargetConfig
from main.core_data.media.audio.config import AudSourceConfig
from main.core_data.media.ecg import EcgTargetConfig
from main.core_data.media.ecg.config import EcgSourceConfig
from main.core_data.media.eeg import EegTargetConfig
from main.core_data.media.eeg.config import EegSourceConfig
from main.core_data.media.text import TxtTargetConfig
from main.core_data.media.text.config import TxtSourceConfig
from main.core_data.media.video import VidTargetConfig
from main.core_data.media.video.config import VidSourceConfig


@dataclasses.dataclass
class DatasetConfig:
    # EEG
    eeg_target_config: Optional[EegTargetConfig] = None
    eeg_source_config: Optional[EegSourceConfig] = None
    # Vid
    vid_target_config: Optional[VidTargetConfig] = None
    vid_source_config: Optional[VidSourceConfig] = None
    # Aud
    aud_target_config: Optional[AudTargetConfig] = None
    aud_source_config: Optional[AudSourceConfig] = None
    # Txt
    txt_target_config: Optional[TxtTargetConfig] = None
    txt_source_config: Optional[TxtSourceConfig] = None
    # ECG
    ecg_target_config: Optional[EcgTargetConfig] = None
    ecg_source_config: Optional[EcgSourceConfig] = None

    unit_seconds: float = 4  # Should depend on EEG model
