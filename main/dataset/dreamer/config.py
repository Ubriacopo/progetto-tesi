import dataclasses

from main.core_data.media.audio.config import AudSourceConfig
from main.core_data.media.eeg.config import EegSourceConfig
from main.core_data.media.text.config import TxtSourceConfig
from main.core_data.media.video.config import VidSourceConfig
from main.dataset.base_config import DatasetConfig


class DreamerEegSourceConfig(EegSourceConfig):
    fs: int = 128


@dataclasses.dataclass
class DreamerConfig(DatasetConfig):
    eeg_source_config: DreamerEegSourceConfig = dataclasses.field(default_factory=DreamerEegSourceConfig)
    aud_source_config: AudSourceConfig = dataclasses.field(default_factory=lambda: AudSourceConfig(fs=44100))
    vid_source_config: VidSourceConfig = dataclasses.field(default_factory=lambda: VidSourceConfig(fps=25))
    txt_source_config: TxtSourceConfig = dataclasses.field(default_factory=TxtSourceConfig)
