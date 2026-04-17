import math
from typing import Optional

import mne
import torch
from cbramod.models.cbramod import CBraMod
from einops import rearrange
from torch import nn

from main.core_data.media.eeg.eeg import EEG
from main.core_data.utils import timed
from main.core_data.media.eeg.channel_canonical_order import EegCanonicalOrder, EegOrder
from main.utils.data import MaskedValue
from main.utils.logging import make_logger


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
            raw.resample(self.tfreq, method="polyphase", npad="auto", verbose=self.verbose)
            return x

        elif isinstance(x, torch.Tensor):
            raw = mne.io.RawArray(x, info=mne.create_info(sfreq=self.sfreq))
            raw.resample(self.tfreq, method="polyphase", npad="auto", verbose=self.verbose)
            return raw.get_data()

        raise NotImplementedError("To call this pipeline you have to use either a Signal or a tensor")


class EEGToTimePatches(nn.Module):
    def __init__(self, points_per_patch: int, max_segments: int):
        """
        Partitions the input EEG in small time patches matching the fs

        :param points_per_patch: How many points are stored inside a patch.
        :param max_segments: Maximum number of time patches that can be extracted
        """
        super().__init__()
        self.points_per_patch = points_per_patch
        self.max_segments = max_segments

        self.max_points = self.points_per_patch * self.max_segments
        self.logger = make_logger(self.__class__.__name__)

    @timed()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor (channels, time), got shape {x.shape}")

        c, d = x.shape
        T = d / self.points_per_patch

        if d == 0:
            raise ValueError("Got empty time axis in EEGToTimePatches")

        # Special case in which the extracted time sequence is longer than allowed
        # (This should never occur)
        if T > self.max_segments:
            # Center crop. Alternative would be sliding window.
            self.logger.warning(f"Warning: Somehow you got more T than allowed ({T} > {self.max_segments}).\n"
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
    def forward(self, x: MaskedValue) -> MaskedValue | torch.Tensor:
        # If masking enabled I expect the mask to be of the shape [c, T].
        data: torch.Tensor = x['data']
        mask: torch.Tensor = x['mask']

        # Optional batch dim (expect batch size 1)
        if self.first_dim_batch and data.ndim == 4:
            if data.shape[0] != 1 or mask.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got data {data.shape[0]}, mask {mask.shape[0]}")
            data, mask = data.squeeze(0), mask.squeeze(0)

        if data.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got {data.shape} but we wanted (c, T, D)")
        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape (c, T), got {mask.shape}")

        c, T, d = data.shape
        if mask.shape != (c, T):
            raise ValueError(f"Mask shape {mask.shape} incompatible with data shape {(c, T, d)}")

        if T > self.max_length:
            # Center crop
            padding = (T - self.max_length) // 2
            data = data[:, padding: padding + self.max_length, :]
            mask = mask[:, padding: padding + self.max_length]

        elif T < self.max_length:
            padding = self.max_length - T
            data = torch.nn.functional.pad(data, (0, 0, 0, padding))  # (C, T_max, D)
            mask = torch.nn.functional.pad(mask, (0, padding))  # (C, T_max)

        # Set time steps first. We get a simpler MASK like this.
        data = rearrange(data, "c t d -> t c d")  # (T, C, D)
        mask = rearrange(mask, "c t -> t c")  # (T, C)
        return data if self.drop_mask else MaskedValue(data=data, mask=mask)


class CBraModEmbedderTransform(nn.Module):
    def __init__(self, weights_path: str, device=None, **kwargs):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = CBraMod(**kwargs).to(self.device)

        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

    @timed()
    def forward(self, x: MaskedValue | torch.Tensor) -> MaskedValue | torch.Tensor:
        mask: Optional[torch.Tensor] = None
        if isinstance(x, dict):
            x, mask = x["data"], x["mask"]

        x: torch.Tensor
        if len(x.shape) == 3:
            # Add the batch
            x = x.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        x = x.float().to(self.device)
        if mask is not None:
            mask: torch.Tensor = mask.bool().to(self.device)

        with torch.inference_mode():
            z = self.model(x=x, mask=~mask if mask is not None else None)

        return z if mask is None else MaskedValue(data=z, mask=mask)


class CanonicalOrderTransform(nn.Module):
    def __init__(self, eeg_order: list[str], canonical_order: EegOrder = EegCanonicalOrder()):
        super().__init__()
        self.canonical_order: EegOrder = canonical_order
        self.eeg_order: list[str] = eeg_order

    def forward(self, x: torch.Tensor) -> MaskedValue:
        x, mask = self.canonical_order.adapt(x, self.eeg_order)
        return MaskedValue(data=x, mask=mask)
