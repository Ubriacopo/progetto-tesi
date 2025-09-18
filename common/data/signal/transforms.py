from typing import Optional

import mne
import torch
from torch import nn

from common.data.signal.signal import Signal
from common.data.transform import IDENTITY


class SignalToTensor(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x: Signal) -> torch.Tensor:
        return torch.from_numpy(x.data.get_data())


class DataAsMneRaw(nn.Module):
    def __init__(self, channel_names: list[str], channel_types: list[str], verbose: bool = False):
        super().__init__()
        self.channel_names = channel_names
        self.channel_types = channel_types
        self.verbose: bool = verbose

    def forward(self, x: Signal) -> Signal:
        # Convertion already happened
        if isinstance(x.data, mne.io.RawArray):
            return x

        if x.data.shape[0] != len(self.channel_names):
            x.data = x.data.T

        info = mne.create_info(ch_names=self.channel_names, ch_types=self.channel_types, sfreq=x.fs)
        raw = mne.io.RawArray(x.data, info=info, verbose=self.verbose)
        x.data = raw

        return x


class SubclipMneRaw(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x: Signal) -> Signal:
        if not isinstance(x.data, mne.io.RawArray):
            raise TypeError("Raw array must be of type mne.io.RawArray")
        tmin, tmax = x.interval
        x.data = x.data.crop(tmin=tmin, tmax=tmax)
        return x


# todo non va bene per ECG
class SignalSequenceResampling(nn.Module):
    def __init__(self, original_fs: int, sequence_duration_seconds: int,
                 resampler: nn.Module = IDENTITY, channels_first: bool = False):
        super().__init__()
        self.sequence_length = original_fs * sequence_duration_seconds
        self.resampler: nn.Module = resampler
        self.channels_first = channels_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_first:
            x = x.T

        segments = int(x.shape[0] / self.sequence_length)
        if x.shape[0] % self.sequence_length != 0:
            segments += 1

        y: Optional[torch.Tensor] = None
        for i in range(segments):
            x_i = x[i * self.sequence_length:(i + 1) * self.sequence_length]
            res = self.resampler(x_i)
            if self.channels_first:
                res = res.T
            # We have new dimension that records the sequence.
            y: torch.Tensor = torch.cat((y, res)) if y is not None else res

        return y
