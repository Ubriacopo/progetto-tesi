import torch
from tensordict import TensorDict
from torch import nn

from main.model.neegavi.model import EegInterAviModel


class SimpleTuneLinearProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int):
        super(SimpleTuneLinearProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone
        self.project = nn.Linear(in_dim, out_dim)
        for p in self.backbone.parameters():
            p.requires_grad = False

        for block in self.backbone.gatedXAttn_layers[-2:]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x: TensorDict) -> torch.Tensor:
        # Data inputs are of the shape [B, B', T, P, D]
        b, b_inner = x.shape[:2]
        x = x.flatten(0, 1)
        y = self.backbone(x)

        # Restore previous batch
        y = y.cls.unflatten(0, (b, b_inner))
        # AVG over N timesteps of a sample
        y = y.mean(dim=-2)
        logits = self.project(y)
        return logits
