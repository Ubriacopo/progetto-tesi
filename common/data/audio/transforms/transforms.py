from typing import Optional

import torch
import torchaudio
from moviepy import AudioFileClip
from torch import nn

from common.data.audio.audio import Audio
from common.data.audio.transforms.utils import check_audio_data
from common.data.transform import IDENTITY


class AudioToTensor(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x: Audio):
        if x.data is None:
            x, waveform = torchaudio.load(x.file_path)
            return x.T  # This is kinda peculiar. I need to pass by torchaudio for 1s clips for some reason (moviepy has issues)

        aud: AudioFileClip = x.data
        x = aud.to_soundarray()
        x = torch.from_numpy(x).float()

        return x


class SubclipAudio(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x: Audio):
        aud: AudioFileClip = x.data
        check_audio_data(x, AudioFileClip)

        x.data = aud.subclipped(x.interval[0], x.interval[1])
        return x


class ToMono(nn.Module):
    """
    Transforms a source from Stereo or any other format to MONO. (Single wave)
    """

    def __init__(self, dim: int = 1, keepdim: bool = False):
        super().__init__()
        self.keepdim: bool = keepdim
        self.dim: int = dim

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected a torch.Tensor")
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)

# todo move in signal and rename
class AudioZeroMasking(nn.Module):
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


class AudioSequenceResampler(nn.Module):
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
