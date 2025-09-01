import torch
from torch import nn
from transformers import VivitImageProcessor, AutoFeatureExtractor


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
        return x


class W2VBertFeatureExtractorTransform(nn.Module):
    def __init__(self, model_name: str = "google/vivit-b-16x2-kinetics400", use_padding: bool = True):
        super().__init__()
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.use_padding = use_padding

    def forward(self, x):
        x = self.processor(x, padding=self.use_padding, return_tensors="pt")
        return x
