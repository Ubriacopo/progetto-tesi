import torch
from torch import nn
from transformers import VivitImageProcessor, AutoFeatureExtractor


# toddo ma va in preprocessing questo?
# todo da metter in embedder zone
class ViVitImageProcessorTransform(nn.Module):
    def __init__(self, model_name: str = "google/vivit-b-16x2-kinetics400"):
        super().__init__()
        self.processor = VivitImageProcessor.from_pretrained(model_name)

    def forward(self, x):
        if isinstance(x, torch.Tensor) and len(x.shape) == 3:
            x = [x]
        elif isinstance(x, torch.Tensor):
            x = list(x.unbind(0))

        x = self.processor(x, return_tensors="pt")
        x["pixel_values"] = x["pixel_values"].squeeze(0)
        return x


class W2VBertFeatureExtractorTransform(nn.Module):
    def __init__(self, model: str = "facebook/w2v-bert-2.0"):
        super().__init__()
        self.extractor = AutoFeatureExtractor.from_pretrained(model)

    def forward(self, x: torch.Tensor):
        o = self.extractor(x, return_tensors="pt", padding=True)
        o["input_features"] = o["input_features"].squeeze()
        o["attention_mask"] = o["attention_mask"].squeeze()
        return o
