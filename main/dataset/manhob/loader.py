import mne

from main.core_data.loader import DataPointsLoader
from main.dataset.manhob.config import ManhobConfig


class ManhobPointsLoader(DataPointsLoader):
    def __init__(self, base_path: str, config: ManhobConfig = ManhobConfig()):
        super().__init__()
        self.base_path: str = base_path
        self.config: ManhobConfig = config

    def scan(self):
        raw = mne.io.read_raw_bdf("recording.bdf", preload=True)
        # todo scan stuff