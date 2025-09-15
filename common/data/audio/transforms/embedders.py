from typing import Optional

import torch
from torch import nn
from transformers import AutoFeatureExtractor, BatchFeature, WavLMModel


class FlattenFeatureExtractorOutput(nn.Module):
    def forward(self, x):
        x.input_values = x.input_values.squeeze()
        if hasattr(x, "attention_mask"):
            x.attention_mask = x.attention_mask.squeeze()
        return x


# Mono audio is supposed.
# todo merge con altro che tanto vanno sempre a braccetto.
class WavLmFeatureExtractorTransform(nn.Module):
    def __init__(self, model_name: str = "microsoft/wavlm-base", sampling_rate: int = None, max_length: int = None):
        super(WavLmFeatureExtractorTransform, self).__init__()
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        # Is already true by default
        self.extractor.do_normalize = True
        self.sampling_fs: int = sampling_rate
        self.max_length = max_length

    # todo refactor che non mi piace cosi
    def forward(self, x: torch.Tensor) -> BatchFeature:
        if len(x.shape) == 2:
            y: Optional[BatchFeature] = None
            # I don't know why but the feature extractor masking fails if I pass batched data.
            for t in x.unbind(0):
                out = self.extractor(t, return_tensors="pt", padding=True,
                                     max_length=self.max_length, sampling_rate=self.sampling_fs)

                if y is None:
                    y = out
                else:
                    y.data["input_values"] = torch.cat((y.input_values, out.input_values))
                    if hasattr(y, "attention_mask"):
                        y.data["attention_mask"] = torch.cat((y.attention_mask, out.attention_mask))

        else:
            y = self.extractor(x, return_tensors="pt", padding=True,
                               max_length=self.max_length, sampling_rate=self.sampling_fs)

        return y


class WavLmEmbedderTransform(nn.Module):
    def __init__(self, model_name: str = "microsoft/wavlm-base", device=None):
        super(WavLmEmbedderTransform, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = WavLMModel.from_pretrained(model_name, device_map=self.device)

    def forward(self, x: BatchFeature) -> torch.Tensor:
        x = x.to(self.device)
        with torch.no_grad():
            y = self.model(**x)
        return y.last_hidden_state


# todo make tests out of this
# Questo fa quello che dovrebbe
if __name__ == "__main__":
    a = torch.randn(16000)
    call = nn.Sequential(
        WavLmFeatureExtractorTransform(),
        WavLmEmbedderTransform(),
    )

    c = call(a)
    print(c.shape)
    b = torch.randn(4, 16000)
    d = call(b)
    print(d.shape)
