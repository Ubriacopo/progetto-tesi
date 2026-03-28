import torch
from tensordict import TensorDict
from torch import nn

from main.model.neegavi.model import EegInterAviModel


class SimpleLinearProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int):
        super(SimpleLinearProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone
        self.project = nn.Linear(in_dim, out_dim)

    def forward(self, x: TensorDict) -> torch.Tensor:
        # Data inputs are of the shape [B, B', T, P, D]
        b_inner = x.shape[1]
        x = x.flatten(0, 1)
        # Frozen encoder
        with torch.inference_mode():
            y = self.backbone(x)

        # Restore previous batch
        y = y.cls.unflatten(0, (-1, b_inner))
        # AVG over N timesteps of a sample
        y = y.mean(dim=-2)
        return self.project(y)
