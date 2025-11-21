import mne
import torch
from torch import nn

from main.core_data.media.signal.signal import Signal


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
            raise TypeError("Raw array must be a mne.io.Arrawy")

        tmin, tmax = x.interval
        tmax = max(min(x.data.duration, tmax), 0)
        tmin = max(min(x.data.duration, tmin), 0)

        x.data = x.data.crop(tmin=tmin, tmax=tmax)
        return x


class BandpassFilter(nn.Module):
    def __init__(self, l_freq=0.5, h_freq=40.0, notch=50.0):
        super().__init__()
        self.lowpass: float = l_freq
        self.highpass: float = h_freq
        self.notch: float = notch

    def forward(self, x: Signal) -> Signal:
        raw: mne.io.RawArray = x.data

        raw.notch_filter(self.notch, verbose=False)
        raw.filter(l_freq=self.lowpass, h_freq=self.highpass, verbose=False, filter_length='auto')

        return x


class SignalZeroMasking(nn.Module):
    def __init__(self, max_length: int | float, fs: int, channels_first: bool = False):
        """

        :param max_length: Expressed in seconds.
        :param fs:
        :param channels_first:
        """
        super().__init__()
        self.fs = fs
        self.max_length = max_length

        self.max_data_points = round(self.max_length * fs)
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
