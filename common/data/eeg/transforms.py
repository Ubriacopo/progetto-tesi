import math
from typing import Optional

import mne
import torch
from cbramod.models.cbramod import CBraMod
from einops import rearrange
from torch import nn

from common.data.eeg.eeg import EEG
from common.data.utils import timed


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


class AddMneAnnotation(nn.Module):
    # noinspection PyMethodMayBeStatic
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

    @timed()
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


class EEGToTimePatches(nn.Module):
    def __init__(self, points_per_patch: int, max_segments: int):
        super().__init__()
        self.points_per_patch = points_per_patch
        self.max_segments = max_segments

        self.max_points = self.points_per_patch * self.max_segments

    @timed()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c, d = x.shape
        T = d / self.points_per_patch

        # Special case in which the extracted time sequence is longer than allowed
        # (This should never occur)
        if T > self.max_segments:
            # Center crop. Alternative would be sliding window.
            print(f"Warning: Somehow you got more T than allowed ({T} > {self.max_segments}).\n"
                  "Center-cropping is applied but investigate if this behaviour is desired.")
            pad = int((d - self.max_points) / 2)
            x = x[:, pad:d - pad]
            x = x[:, :self.max_points]  # To be sure we took the correct number of points
            x = rearrange(x, "c (t d) -> c t d", d=self.points_per_patch)
            return x

        next_multiple = math.ceil(d / self.points_per_patch) * self.points_per_patch
        missing_points = next_multiple - d
        if missing_points != 0:
            # We have to pad the last one
            x = torch.nn.functional.pad(x, (0, missing_points))

        x = rearrange(x, 'c (t d) -> c t d', d=self.points_per_patch)
        return x


class EegTimePadding(nn.Module):
    def __init__(self, max_length: int, drop_mask: bool = False, first_dim_batch: bool = True):
        super().__init__()
        self.max_length: int = max_length
        self.first_dim_batch: bool = first_dim_batch  # When the dim is batch but always 1
        self.drop_mask: bool = drop_mask

    @timed()
    def forward(self, x: torch.Tensor) -> dict | torch.Tensor:
        if self.first_dim_batch:
            x = x.squeeze(0)

        if not len(x.shape) == 3:
            raise ValueError(f"Expected 3D tensor, got {x.shape}. We want (c, T, D)")

        T = x.shape[-2]
        # Channels x T (Easier to use later).
        mask = torch.zeros(self.max_length).bool()
        if self.max_length > T:
            x = torch.nn.functional.pad(x, (0, 0, 0, self.max_length - T))
            # Set time steps first. We get a simpler MASK like this.
            x = rearrange(x, 'c t d -> t c d')
            mask[:T] = True  # Zeros

        return {"data": x, "mask": mask} if not self.drop_mask else x


class CBraModEmbedderTransform(nn.Module):
    def __init__(self, weights_path: str, device=None, **kwargs):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = CBraMod(**kwargs).to(device)

        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    @timed()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add the batch

        with torch.inference_mode():
            # I don't know what is wrong with CBraMod. I made a mistake somewhere.
            z = self.model(x.float().to("cpu"))
        return z
