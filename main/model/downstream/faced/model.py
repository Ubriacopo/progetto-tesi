from typing import TypedDict

import torch
from torch import nn

from main.model.neegavi.model import EegInterAviModel
from main.utils.data import MaskedValue


class FacedInput(TypedDict):
    # Faced only has EEG input
    eeg: MaskedValue


class FacedLinearProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int):
        super(FacedLinearProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone
        self.head: nn.Module = nn.Linear(in_dim, out_dim)

    def forward(self, x: FacedInput) -> torch.Tensor:
        y = self.backbone(x)  # TODO eeginteravi inference only model not all
        cls_token = y.cls
        # How to fuse things todo

        # todo mean poioling over tokens (sono n)

        return self.head(cls_token)
