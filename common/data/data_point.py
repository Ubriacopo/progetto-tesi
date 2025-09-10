from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Optional, Any, Iterable

import torch
from torch import nn

from common.data import media_types
from common.data.audio import Audio
from common.data.eeg import EEG
from common.data.media import Media
from common.data.text import Text
from common.data.video import Video


class DatasetDataPoint(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @staticmethod
    @abstractmethod
    def from_dict(o: dict) -> DatasetDataPoint:
        pass

    @staticmethod
    @abstractmethod
    def get_identifier() -> str:
        pass


# TODO: Se aggiungo info questa struttura non mi basta.
@dataclasses.dataclass
class EEGDatasetDataPoint(DatasetDataPoint):
    @staticmethod
    def from_dict(o: dict, base_path: str = None) -> EEGDatasetDataPoint:
        return EEGDatasetDataPoint(
            entry_id=o["entry_id"],
            eeg=EEG.restore_from_dict(o, base_path),
            vid=Video.restore_from_dict(o, base_path),
            aud=Audio.restore_from_dict(o, base_path),
            txt=Text.restore_from_dict(o, base_path),
        )

    @staticmethod
    def get_identifier() -> str:
        return "entry_id"

    entry_id: str  # We suppose every entry has a unique way of identifying itself
    # EEG dataset supposes to have for an entry a list of recordings. Other data types are optional.
    eeg: EEG | torch.Tensor

    # TODO: What about multiple medias? (front-rgp-etc). -> sarebbe array di Video etc
    #       Da fare dopo ora ci va bene così
    vid: Optional[Video | torch.Tensor] = None
    txt: Optional[Text | torch.Tensor] = None
    aud: Optional[Audio | torch.Tensor] = None
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        d = {"entry_id": self.entry_id, } | self.eeg.to_dict()
        if self.vid is not None:
            d = d | self.vid.to_dict()
        if self.txt is not None:
            d = d | self.txt.to_dict()
        if self.aud is not None:
            d = d | self.aud.to_dict()
        return d


class EEGDatasetTransformWrapper:
    @staticmethod
    def init_transform(transform: nn.Sequential | list | None) -> nn.Sequential | None:
        if transform is None:
            return None
        if isinstance(transform, Iterable):
            transform = nn.Sequential(*transform)
        return transform

    def __init__(self, name: str,
                 eeg_transform: nn.Sequential | Iterable[nn.Module] | None = None,
                 vid_transform: nn.Sequential | Iterable[nn.Module] | None = None,
                 aud_transform: nn.Sequential | Iterable[nn.Module] | None = None,
                 txt_transform: nn.Sequential | Iterable[nn.Module] | None = None,
                 xmod_first: bool = False,
                 xmod_transform: nn.Sequential | Iterable[nn.Module] | None = None, ):
        self.name: str = name

        self.eeg_transform: Optional[nn.Sequential] = self.init_transform(eeg_transform)
        self.vid_transform: Optional[nn.Sequential] = self.init_transform(vid_transform)
        self.aud_transform: Optional[nn.Sequential] = self.init_transform(aud_transform)
        self.txt_transform: Optional[nn.Sequential] = self.init_transform(txt_transform)

        self.xmod_first: bool = xmod_first
        self.xmod_transform: Optional[nn.Sequential] = self.init_transform(xmod_transform)


def call_pipelines(x: EEGDatasetDataPoint, pipe_wrapper: EEGDatasetTransformWrapper) -> EEGDatasetDataPoint:
    # Cross modality first if defined and wanted
    if pipe_wrapper.xmod_transform is not None and pipe_wrapper.xmod_first:
        x = pipe_wrapper.xmod_transform(x)

    if pipe_wrapper.eeg_transform is not None:
        x.eeg = pipe_wrapper.eeg_transform(x.eeg)
    if pipe_wrapper.vid_transform is not None:
        x.vid = pipe_wrapper.vid_transform(x.vid)
    if pipe_wrapper.aud_transform is not None:
        x.aud = pipe_wrapper.aud_transform(x.aud)
    if pipe_wrapper.txt_transform is not None:
        x.txt = pipe_wrapper.txt_transform(x.txt)

    if pipe_wrapper.xmod_transform is not None and not pipe_wrapper.xmod_first:
        x = pipe_wrapper.xmod_transform(x)

    return x


class AgnosticDatasetPoint(DatasetDataPoint):
    def __init__(self, eid: str | int, *modality: tuple[str, dict | Media]):
        self.eid = eid
        for (k, o) in modality:
            self.__setattr__(k, o)

    def to_dict(self) -> dict:
        o = {self.get_identifier(): self.eid}
        for attr, value in self.__dict__.items():
            if isinstance(value, Media) or hasattr(value, "to_dict_new"):
                # Custom dict logic. Should add classname to it for restore?
                value = value.to_dict_new()

            o |= {attr: value}

        return o

    def __getitem__(self, item: str):
        return getattr(self, item)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    @staticmethod
    def get_identifier() -> str:
        return "eid"

    @staticmethod
    def from_dict(o: dict, base_path: str = None) -> AgnosticDatasetPoint:
        objects = []
        for attr, value in o.items():
            # Only exception we can handle inside the DatasetPoint
            if attr == AgnosticDatasetPoint.get_identifier():
                continue

            assert isinstance(value, dict)
            # Path of restoring the object of type else it is just dict
            if "classname" in value:
                media_type: Media = getattr(media_types, value["classname"])
                objects.append((attr, media_type.restore_from_dict_new(value)))
            else:
                objects.append((attr, value))
        # Flexible to any new structure
        return AgnosticDatasetPoint(o["eid"], *objects)

    def export(self, base_path: str = None, *exceptions: str, only: str = None):
        # Only one execution branch
        if only is not None and only in self.__dict__.items():
            attr = self.__getattribute__(only)
            if isinstance(attr, Media) or hasattr(attr, "export"):
                attr.export(base_path)
                return

        # All except the exceptions
        for attr, value in self.__dict__.items():
            if not isinstance(value, Media) and not hasattr(value, "export"):
                continue

            # Ignore the currently processed element.
            if attr in exceptions:
                continue

            value.export(base_path)


class AgnosticDatasetTransformWrapper:
    def __init__(self, name: str, *transforms: tuple[str, nn.Sequential]):
        self.name = name
        for (k, o) in transforms:
            self.__setattr__(k, o)

    def __getitem__(self, item: str):
        return getattr(self, item)

    def is_defined(self, item: str):
        return item in self.__dict__

    def call(self, x: AgnosticDatasetPoint):
        for key, value in x.__dict__.items():
            if self.is_defined(key):
                # Call each transform that maps to x definition
                x[key] = self[key](value)
