import dataclasses
from abc import ABC, abstractmethod
from typing import Optional

from common.data.audio import Audio
from common.data.eeg import EEG
from common.data.text import Text
from common.data.video import Video


class DatasetDataPoint(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def get_identifier() -> str:
        pass


@dataclasses.dataclass
class EEGDatasetDataPoint(DatasetDataPoint):
    @staticmethod
    def get_identifier() -> str:
        return "entry_id"

    entry_id: str  # We suppose every entry has a unique way of identifying itself
    # EEG dataset supposes to have for an entry a list of recordings. Other data types are optional.
    eeg: EEG
    # TODO: What about multiple medias? (front-rgp-etc). -> sarebbe array di Video etc
    #       Da fare dopo ora ci va bene così
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
