import torch
from torch import nn

from main.model.neegavi.model import EegInterAviModel


class FacedLinearProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int):
        super(FacedLinearProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone
        self.head: nn.Module = nn.Linear(in_dim, out_dim)

    def forward(self, x: dict) -> torch.Tensor:
        y = self.backbone(x)  # TODO eeginteravi inference only model not all
        y = y.cls
        # How to fuse things todo
        return self.head(y)
