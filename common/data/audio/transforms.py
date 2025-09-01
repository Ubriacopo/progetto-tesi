from typing import Optional

import torch
from moviepy import AudioFileClip
from torch import nn

from .audio import Audio


def check_audio_data(x, data_type: type):
    if not isinstance(x, Audio):
        raise TypeError("Given object is not of required type Audio")

    if x.data is None:
        raise ValueError("Audio has to be loaded first.")

    if not isinstance(x.data, data_type):
        raise TypeError("Given audio object is not valid")


class AudioToTensor(nn.Module):
    def forward(self, x: Audio):
        aud: Optional[AudioFileClip] = x.data
        if x.data is None:
            aud = AudioFileClip(x.file_path)

        x = aud.to_soundarray()
        x = torch.from_numpy(x).float()

        return x


class SubclipAudio(nn.Module):
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

        self.keepdim = keepdim
        self.dim = dim

    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected a torch.Tensor")
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)
