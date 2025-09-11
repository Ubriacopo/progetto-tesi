from typing import Optional

import mne
import torch
from einops import rearrange
from torch import nn

from .eeg import EEG
from .mne_utils import find_segment_by_descriptor


class EEGToTensor(nn.Module):
    def __init__(self, take_eeg: bool = True, take_ecg: bool = False):
        super().__init__()
        self.take_eeg: bool = take_eeg
        self.take_ecg: bool = take_ecg

    def forward(self, x: EEG) -> torch.Tensor:
        raw: Optional[mne.io.BaseRaw] = x.data
        if raw is None:
            fif = mne.io.read_raw_fif(x.file_path)
            segments = find_segment_by_descriptor(fif, x.eid)
            if len(segments) == 0 or len(segments) > 1:
                raise RuntimeError(f"Found {len(segments)} EEG for {x.eid}. There an error with {x.eid}.")
            _, onset, duration = segments[0]
            raw = fif.copy().crop(tmin=onset, tmax=onset + duration)

        picks = mne.pick_types(raw.info, eeg=self.take_eeg, ecg=self.take_ecg)
        x = torch.from_numpy(raw.get_data(picks))

        return x


class EEGToMneRawFromChannels(nn.Module):
    def __init__(self, channel_names: list[str], channel_types: list[str], verbose: bool = False):
        super().__init__()
        self.channel_names: list[str] = channel_names
        self.channel_types: list[str] = channel_types
        self.verbose: bool = verbose

    # todo refactor
    def forward(self, x: EEG) -> EEG:
        if x.data is None:
            fif = mne.io.read_raw_fif(x.file_path)
            segments = find_segment_by_descriptor(fif, x.eid)
            if len(segments) == 0 or len(segments) > 1:
                raise RuntimeError(f"Found {len(segments)} EEG for {x.eid}. There an error with {x.eid}.")
            _, onset, duration = segments[0]
            raw = fif.copy().crop(tmin=onset, tmax=onset + duration)
            x.data = raw
            return x

        info = mne.create_info(ch_names=self.channel_names, ch_types=self.channel_types, sfreq=x.fs)
        raw = mne.io.RawArray(x.data, info=info, verbose=self.verbose)
        x.data = raw
        return x


class EEGMneAddAnnotation(nn.Module):
    def forward(self, x: EEG):
        raw: mne.io.BaseRaw = x.data
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError("To call this pipeline you have to turn to MNE object first ")

        if x.eid is None:
            raise TypeError("A valid descriptor to identify the annotation is required")

        start, stop = x.interval
        new_annotation = mne.Annotations(onset=[start], duration=[stop - start], description=x.eid, orig_time=None)
        existing = getattr(raw, 'annotations', None)

        merged = new_annotation if existing is None else existing + new_annotation
        raw.set_annotations(merged)
        return x


class EEGResample(nn.Module):
    def __init__(self, target_fs: int, sfreq: int = None, verbose: bool = False):
        super().__init__()
        self.sfreq = sfreq
        self.fs = target_fs
        self.verbose = verbose

    def forward(self, x: EEG | torch.Tensor) -> EEG | torch.Tensor:
        if isinstance(x, EEG):
            raw: mne.io.RawArray = x.data
            # TODO: Tweak this call to be good.
            raw.resample(self.fs, method="polyphase", npad="auto", verbose=self.verbose)
            return x

        elif isinstance(x, torch.Tensor):
            raw = mne.io.RawArray(x, info=mne.create_info(sfreq=self.sfreq))
            # TODO: Tweak this call to be good.
            raw.resample(self.fs, method="polyphase", npad="auto", verbose=self.verbose)
            return raw.get_data()

        raise NotImplementedError("To call this pipeline you have to turn to MNE object or Tensor first")


class EEGToTimePatches(nn.Module):
    def __init__(self, points_per_patch: int, max_segments: int = 8):
        super().__init__()
        self.points_per_patch = points_per_patch
        self.max_segments = max_segments

        self.max_points = self.points_per_patch * self.max_segments

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, d = x.shape
        # [16000] -> [T, points_per_patch]
        T = d / self.points_per_patch

        if T > self.max_segments:
            # Center crop. Alternative would be sliding window, but we would be moving towards
            # the model (CBraMod needs to be involved or something else)
            pad = int((d - self.max_points) / 2)
            x = x[:, pad:d - pad]
            x = x[:, :self.max_points]  # To be sure we took the correct number of points
            x = rearrange(x, "c (t d) -> c t d", t=self.max_segments)

        if T == self.max_segments:
            x = rearrange(x, "c (t d) -> c t d", t=self.max_segments)

        if T < self.max_segments:
            # Do zero padding. TODO See if really so for CBraMod
            x = torch.cat([x, torch.zeros(c, self.max_points - d)], dim=1)
            x = rearrange(x, "c (t d) -> c t d", t=self.max_segments)
            return x

        return x
