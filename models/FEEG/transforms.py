import torch
from torch import nn
from transformers import AutoFeatureExtractor


# toddo ma va in preprocessing questo?
# todo da metter in embedder zone
# todo visionare bene con time sequences.


class W2VBertFeatureExtractorTransform(nn.Module):
    def __init__(self, model: str = "facebook/w2v-bert-2.0", force_time_seq: bool = False):
        super().__init__()
        self.extractor = AutoFeatureExtractor.from_pretrained(model)
        self.force_time_seq = force_time_seq

    def forward(self, x: torch.Tensor):
        o = self.extractor(x, return_tensors="pt", padding=True)

        if not self.force_time_seq:
            o["input_features"] = o["input_features"].squeeze()
            o["attention_mask"] = o["attention_mask"].squeeze()

        return o
