from typing import Optional

import numpy as np
import torch
import torchaudio
from moviepy import AudioFileClip
from torch import nn
from transformers import AutoFeatureExtractor

from .audio import Audio
from ..transform import SequenceResampler, IDENTITY


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
            res = res.unsqueeze(0)
            y: torch.Tensor = torch.cat((y, res)) if y is not None else res

        return y


class ComputeFeatureHubert(nn.Module):
    def __init__(self, original_fs: int):
        super().__init__()
        self.original_fs = original_fs

    def forward(self, x: torch.Tensor):
        return torchaudio.functional.resample(x, self.original_fs, torchaudio.pipelines.HUBERT_BASE.sample_rate)


# todo visionare bene con time sequences.
class W2VBertFeatureExtractorTransform(nn.Module):
    def __init__(self, model: str = "facebook/w2v-bert-2.0", force_time_seq: bool = False):
        super().__init__()
        self.extractor = AutoFeatureExtractor.from_pretrained(model)
        self.force_time_seq = force_time_seq

    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unbind(0)

        o = self.extractor(x, return_tensors="pt", padding=True)
        if not self.force_time_seq:
            o["input_features"] = o["input_features"].squeeze()
            o["attention_mask"] = o["attention_mask"].squeeze()

        return o


class HubertBaseFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torchaudio.pipelines.HUBERT_BASE.get_model().to(device)

    def forward(self, x: torch.Tensor):
        y, _ = self.model.extract_features(x)
        y = y[-1][0].mean(0)  # TODO vedi non so bene cosa faccia.
        return y
