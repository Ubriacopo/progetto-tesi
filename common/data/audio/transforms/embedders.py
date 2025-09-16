from typing import Optional

import torch
import torchaudio
from torch import nn
from transformers import AutoFeatureExtractor, BatchFeature, WavLMModel


class FlattenFeatureExtractorOutput(nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        x.input_values = x.input_values.squeeze()
        if hasattr(x, "attention_mask"):
            x.attention_mask = x.attention_mask.squeeze()
        return x


class WavLm:
    # Mono audio is supposed.
    # todo merge con altro che tanto vanno sempre a braccetto.
    class FeatureExtractorTransform(nn.Module):
        def __init__(self, model_name: str = "microsoft/wavlm-base", sampling_rate: int = None, max_length: int = None):
            super(WavLm.FeatureExtractorTransform, self).__init__()
            self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
            # Is already true by default
            self.extractor.do_normalize = True
            self.sampling_fs: int = sampling_rate
            self.max_length = max_length

        # todo refactor che non mi piace cosi
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

    class EmbedderTransform(nn.Module):
        def __init__(self, model_name: str = "microsoft/wavlm-base", device=None):
            super(WavLm.EmbedderTransform, self).__init__()
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
            self.model = WavLMModel.from_pretrained(model_name, device_map=self.device)

        def forward(self, x: BatchFeature) -> torch.Tensor:
            x = x.to(self.device)
            with torch.no_grad():
                y = self.model(**x)
            return y.last_hidden_state


class HubertBase:
    """
    Hubert relative steps.
    """

    class ComputeFeatureHubert(nn.Module):
        def __init__(self, original_fs: int):
            super().__init__()
            self.original_fs = original_fs

        def forward(self, x: torch.Tensor):
            return torchaudio.functional.resample(x, self.original_fs, torchaudio.pipelines.HUBERT_BASE.sample_rate)

    class FeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = torchaudio.pipelines.HUBERT_BASE.get_model().to(device)

        def forward(self, x: torch.Tensor):
            y, _ = self.model.extract_features(x)
            y = y[-1][0].mean(0)  # TODO vedi non so bene cosa faccia.
            return y


class W2VBert:
    class FeatureExtractorTransform(nn.Module):
        def __init__(self, model: str = "facebook/w2v-bert-2.0", force_time_seq: bool = False):
            super(W2VBert.FeatureExtractorTransform, self).__init__()
            self.extractor = AutoFeatureExtractor.from_pretrained(model)
            self.force_time_seq = force_time_seq

        def forward(self, x: torch.Tensor) -> BatchFeature:
            if len(x.shape) == 3:
                x = x.unbind(0)

            features = self.extractor(x, return_tensors="pt", padding=True)
            if not self.force_time_seq:
                features["input_features"] = features["input_features"].squeeze()
                features["attention_mask"] = features["attention_mask"].squeeze()
            return features


# todo make tests out of this
# Questo fa quello che dovrebbe
if __name__ == "__main__":
    a = torch.randn(16000)
    call = nn.Sequential(
        WavLm.FeatureExtractorTransform(),
        WavLm.EmbedderTransform(),
    )

    c = call(a)
    print(c.shape)
    b = torch.randn(4, 16000)
    d = call(b)
    print(d.shape)
