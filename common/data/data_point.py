from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Optional, Any

import torch
from torch import nn

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
    def from_dict(o: dict) -> DatasetDataPoint:
        pass

    @staticmethod
    @abstractmethod
    def get_identifier() -> str:
        pass


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


# todo circ dependency? e rename
@dataclasses.dataclass
class EEGModalityComposeWrapper:
    eeg_transform: nn.Sequential | None = None
    vid_transform: nn.Sequential | None = None
    aud_transform: nn.Sequential | None = None
    txt_transform: nn.Sequential | None = None
    # Cross Modality Transform. Run after others by default. Not used yet
    xmod_transform: nn.Sequential | None = None
    xmod_first: bool = False


def call_pipelines(x: EEGDatasetDataPoint, pipe_wrapper: EEGModalityComposeWrapper) -> EEGDatasetDataPoint:
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
