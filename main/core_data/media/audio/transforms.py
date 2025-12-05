from typing import Optional

import torch
import torchaudio
from moviepy import AudioFileClip
from torch import nn
from transformers import AutoFeatureExtractor, BatchFeature, WavLMModel

from main.core_data.media.audio.audio import Audio
from main.core_data.processing.transform import IDENTITY
from main.core_data.utils import timed, call_log
from main.utils.logging import make_logger


class AudioToTensor(nn.Module):
    def __init__(self, map_to=None):
        super().__init__()
        self.map_to = map_to

    # noinspection PyMethodMayBeStatic
    @timed()
    @call_log()
    def forward(self, x: Audio):
        aud: AudioFileClip = x.data
        x = aud.to_soundarray()
        x = torch.from_numpy(x).float()

        if self.map_to is not None:
            x = x.to(self.map_to)

        aud.close()  # Close the process we are done with it
        return x


class SubclipAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.logger = make_logger(self.__class__.__name__)

    @timed()
    @call_log()
    def forward(self, x: Audio):
        x.data = AudioFileClip(x.filepath)
        aud: AudioFileClip = x.data

        if x.fs != aud.fps:
            self.logger.error(f"fs mismatch (actual/stored): {aud.fps} != {x.fs}")

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

    @call_log()
    def forward(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(x)}")
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)


class MonoAudioSequencePartitioning(nn.Module):
    def __init__(self, fs: int, sequence_duration_seconds: float, resampler: nn.Module = IDENTITY):
        super().__init__()
        self.sequence_length = round(fs * sequence_duration_seconds)
        self.resampler: nn.Module = resampler

    @call_log()
    @timed()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        segments = int(x.shape[0] / self.sequence_length)
        if x.shape[0] % self.sequence_length != 0:
            segments += 1

        y: Optional[torch.Tensor] = None
        for i in range(segments):
            x_i = x[i * self.sequence_length:(i + 1) * self.sequence_length]
            res = self.resampler(x_i)
            # We have new dimension that records the sequence.
            y: torch.Tensor = torch.cat((y, res)) if y is not None else res

        return y


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
    def __init__(self, model_name: str = "microsoft/wavlm-base", device=None, map_to=None):
        super(WavLmEmbedderTransform, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = WavLMModel.from_pretrained(model_name, device_map=self.device)
        self.map_to = map_to

    @timed()
    def forward(self, x: BatchFeature) -> torch.Tensor:
        x = x.to(self.device)
        with torch.no_grad():
            y = self.model(**x)
        y = y.last_hidden_state
        if self.map_to is not None:
            y = y.to(self.map_to)
        return y


class HubertBaseComputeFeature(nn.Module):
    def __init__(self, original_fs: int):
        super().__init__()
        self.original_fs = original_fs

    @timed()
    def forward(self, x: torch.Tensor):
        return torchaudio.functional.resample(x, self.original_fs, torchaudio.pipelines.HUBERT_BASE.sample_rate)


class HubertFeatureExtractor(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = torchaudio.pipelines.HUBERT_BASE.get_model()
        self.model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            x = x.unsqueeze(0)

        with torch.inference_mode():
            x = x.to(self.device)
            item_audio, _ = self.model.extract_features(x)
            item_audio = item_audio[-1][0].mean(0)

        return item_audio


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
