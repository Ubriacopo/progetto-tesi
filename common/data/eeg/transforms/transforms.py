from typing import Optional

import mne
import torch
from einops import rearrange
from torch import nn

from common.data.eeg.eeg import EEG
from common.data.eeg.mne_utils import find_segment_by_descriptor


class EEGToTensor(nn.Module):
    def __init__(self, take_eeg: bool = True, take_ecg: bool = False):
        super().__init__()
        self.take_eeg: bool = take_eeg
        self.take_ecg: bool = take_ecg

    def forward(self, x: EEG) -> torch.Tensor:
        raw: Optional[mne.io.BaseRaw] = x.data
        picks = mne.pick_types(raw.info, eeg=self.take_eeg, ecg=self.take_ecg)
        x = torch.from_numpy(raw.get_data(picks))
        return x


class EEGDataAsMneRaw(nn.Module):
    def __init__(self, channel_names: list[str], channel_types: list[str], verbose: bool = False):
        super().__init__()
        self.channel_names: list[str] = channel_names
        self.channel_types: list[str] = channel_types
        self.verbose: bool = verbose

    def forward(self, x: EEG) -> EEG:
        if x.data.shape[0] != len(self.channel_names):
            x.data = x.data.T
        info = mne.create_info(ch_names=self.channel_names, ch_types=self.channel_types, sfreq=x.fs)
        raw = mne.io.RawArray(x.data, info=info, verbose=self.verbose)
        x.data = raw
        return x


class AddMneAnnotation(nn.Module):
    def forward(self, x: EEG):
        raw: mne.io.BaseRaw = x.data
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError("To call this pipeline you have to turn to MNE object first ")
        if x.eid is None:
            raise ValueError("A valid descriptor to identify the annotation is required")

        start, stop = x.interval
        new_annotation = mne.Annotations(onset=[start], duration=[stop - start], description=x.eid, orig_time=None)
        existing = getattr(raw, 'annotations', None)

        merged = new_annotation if existing is None else existing + new_annotation
        raw.set_annotations(merged)
        return x


class EEGResample(nn.Module):
    def __init__(self, tfreq: int, sfreq: int = None, verbose: bool = False):
        super().__init__()
        self.sfreq = sfreq
        self.tfreq = tfreq
        self.verbose = verbose

    def forward(self, x: EEG | torch.Tensor) -> EEG | torch.Tensor:
        if isinstance(x, EEG):
            raw: mne.io.RawArray = x.data
            # TODO: Tweak this call to be good.
            raw.resample(self.tfreq, method="polyphase", npad="auto", verbose=self.verbose)
            return x

        elif isinstance(x, torch.Tensor):
            raw = mne.io.RawArray(x, info=mne.create_info(sfreq=self.sfreq))
            # TODO: Tweak this call to be good.
            raw.resample(self.tfreq, method="polyphase", npad="auto", verbose=self.verbose)
            return raw.get_data()

        raise NotImplementedError("To call this pipeline you have to turn to MNE object or Tensor first")
