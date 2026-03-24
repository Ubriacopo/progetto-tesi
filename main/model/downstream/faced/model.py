from typing import TypedDict

import torch
from tensordict import TensorDict
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

    def forward(self, x: TensorDict) -> torch.Tensor:
        b_inner = x.shape[1]
        x = x.flatten(0, 1)
        with torch.no_grad():
            y = self.backbone(x)  # TODO eeginteravi inference only model not all
        # Restore batch
        y = y.cls.unflatten(0, (-1, b_inner))
        # Pool
        y = y.mean(dim=-2)
        return self.head(y)
