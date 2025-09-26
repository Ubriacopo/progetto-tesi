from typing import Optional

import torch
import torchaudio
from moviepy import AudioFileClip
from torch import nn
from transformers import AutoFeatureExtractor, BatchFeature, WavLMModel

from common.data.audio import Audio
from common.data.audio.audio import Audio
from common.data.transform import IDENTITY
from common.data.utils import timed


class AudioToTensor(nn.Module):
    # noinspection PyMethodMayBeStatic
    @timed()
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
    @timed()
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


class AudioSequencePartitioning(nn.Module):
    def __init__(self, fs: int, sequence_duration_seconds: int,
                 resampler: nn.Module = IDENTITY, channels_first: bool = False):
        super().__init__()
        self.sequence_length = round(fs * sequence_duration_seconds)
        self.resampler: nn.Module = resampler
        self.channels_first = channels_first

    @timed()
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


def check_audio_data(x, data_type: type):
    if not isinstance(x, Audio):
        raise TypeError("Given object is not of required type Audio")

    if x.data is None:
        raise ValueError("Audio has to be loaded first.")

    if not isinstance(x.data, data_type):
        raise TypeError("Given audio object is not valid")


class FlattenFeatureExtractorOutput(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        x.input_values = x.input_values.squeeze()
        if hasattr(x, "attention_mask"):
            x.attention_mask = x.attention_mask.squeeze()
        return x


class WavLmFeatureExtractorTransform(nn.Module):
    def __init__(self, model_name: str = "microsoft/wavlm-base", sampling_rate: int = None, max_length: int = None):
        super(WavLmFeatureExtractorTransform, self).__init__()
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        # Is already true by default
        self.extractor.do_normalize = True
        self.sampling_fs: int = sampling_rate
        self.max_length = max_length

    @timed()
    def forward(self, x: torch.Tensor) -> BatchFeature:
        fs = self.sampling_fs
        if len(x.shape) != 2:
            y = self.extractor(x, return_tensors="pt", padding=True, max_length=self.max_length, sampling_rate=fs)
            return y

        y: Optional[BatchFeature] = None
        for t in x.unbind(0):
            out = self.extractor(t, return_tensors="pt", padding=True, max_length=self.max_length, sampling_rate=fs)

            if y is None:
                y = out  # First init
            else:
                y.data["input_values"] = torch.cat((y.input_values, out.input_values))
                if hasattr(y, "attention_mask"):
                    y.data["attention_mask"] = torch.cat((y.attention_mask, out.attention_mask))

        return y


class WavLmEmbedderTransform(nn.Module):
    def __init__(self, model_name: str = "microsoft/wavlm-base", device=None):
        super(WavLmEmbedderTransform, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = WavLMModel.from_pretrained(model_name, device_map=self.device)

    @timed()
    def forward(self, x: BatchFeature) -> torch.Tensor:
        x = x.to(self.device)
        with torch.no_grad():
            y = self.model(**x)
        return y.last_hidden_state


class HubertBaseComputeFeature(nn.Module):
    def __init__(self, original_fs: int):
        super().__init__()
        self.original_fs = original_fs

    @timed()
    def forward(self, x: torch.Tensor):
        return torchaudio.functional.resample(x, self.original_fs, torchaudio.pipelines.HUBERT_BASE.sample_rate)


class W2VBertFeatureExtractorTransform(nn.Module):
    def __init__(self, model: str = "facebook/w2v-bert-2.0", force_time_seq: bool = False):
        super(W2VBertFeatureExtractorTransform, self).__init__()
        self.extractor = AutoFeatureExtractor.from_pretrained(model)
        self.force_time_seq = force_time_seq

    @timed()
    def forward(self, x: torch.Tensor) -> BatchFeature:
        if len(x.shape) == 3:
            x = x.unbind(0)

        features = self.extractor(x, return_tensors="pt", padding=True)
        if not self.force_time_seq:
            features["input_features"] = features["input_features"].squeeze()
            features["attention_mask"] = features["attention_mask"].squeeze()
        return features
