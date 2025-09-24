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
        # todo copy signal!
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


class SignalZeroMasking(nn.Module):
    def __init__(self, max_length: int, fs: int, channels_first: bool = False):
        """

        :param max_length: Expressed in seconds.
        :param fs:
        :param channels_first:
        """
        super().__init__()
        self.fs = fs
        self.max_length = max_length

        self.max_data_points = self.max_length * fs
        self.channels_first = channels_first

    def forward(self, x: torch.Tensor):
        transposed = False
        if len(x.shape) == 2 and not self.channels_first:
            transposed = True
            x = x.T

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        x_points = x.shape[-1]
        if x_points > self.max_data_points:
            # Truncate
            pad = int((x_points - self.max_data_points) / 2)
            x = x[:, pad:x_points - pad]
            x = x[:, :self.max_data_points]
            return x if not transposed else x.T

        if x_points == self.max_data_points:
            return x if not transposed else x.T

        if x_points < self.max_data_points:
            x = torch.cat([x, torch.zeros(x.shape[0], self.max_data_points - x_points)], dim=-1)
            return x if not transposed else x.T

        raise ValueError("Somehow you got here how can that be!")
