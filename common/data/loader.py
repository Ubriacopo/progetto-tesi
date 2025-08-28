import dataclasses
from abc import abstractmethod, ABC
from typing import Optional, Generator, Iterator

from common.data.audio.audio import Audio
from common.data.eeg.eeg import EEG
from common.data.text.text import Text
from common.data.video.video import Video


@dataclasses.dataclass
class EEGDatasetDataCollection:
    entry_id: str  # We suppose every entry has a unique way of identifying itself
    # EEG dataset supposes to have for an entry a list of recordings. Other data types are optional.
    eeg: EEG
    vid: Optional[Video] = None
    txt: Optional[Text] = None
    aud: Optional[Audio] = None

    def to_dict(self) -> dict:
        d = {"entry_id": self.entry_id, } | self.eeg.to_dict()
        if self.vid is not None:
            d = d | self.vid.to_dict()
        if self.txt is not None:
            d = d | self.txt.to_dict()
        if self.aud is not None:
            d = d | self.aud.to_dict()
        return d


class DataLoader(ABC):
    """
    Loads samples of a dataset as reference points.
    """

    @abstractmethod
    def scan(self) -> Iterator[EEGDatasetDataCollection]:
        pass
