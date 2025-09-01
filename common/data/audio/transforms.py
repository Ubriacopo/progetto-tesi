from typing import Optional

import numpy as np
import torch
import torchaudio
from moviepy import AudioFileClip
from torch import nn
from transformers import AutoFeatureExtractor

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
        if x.data is None:
            x, waveform = torchaudio.load(x.file_path)
            return x.T  # This is kinda peculiar. I need to pass by torchaudio for 1s clips for some reason (moviepy has issues)

        aud: AudioFileClip = x.data
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


class AudioZeroMasking(nn.Module):
    def __init__(self, max_length: int, fs: int, channels_first: bool = False):
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
            x = torch.cat([x, np.zeros(self.max_data_points - x_points)], dim=-1)
            return x if not transposed else x.T


# TODO: Valutare dove questo va nel modello :(
#       Sembra sensato dataloader ma per strutture boh
class W2VBertFeatureExtractorTransform(nn.Module):
    def __init__(self, model: str = "facebook/w2v-bert-2.0"):
        super().__init__()
        self.extractor = AutoFeatureExtractor.from_pretrained(model)

    def forward(self, x: torch.Tensor):
        o = self.extractor(x, return_tensors="pt", padding=True)
        o["input_features"] = o["input_features"].squeeze()
        o["attention_mask"] = o["attention_mask"].squeeze()
        return o
