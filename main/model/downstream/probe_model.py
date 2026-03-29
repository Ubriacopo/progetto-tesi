import torch
from tensordict import TensorDict
from torch import nn

from main.model.neegavi.model import EegInterAviModel


class SimpleCbraLinearProbe(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(SimpleCbraLinearProbe, self).__init__()
        self.project = nn.Linear(in_dim, out_dim)

    def forward(self, batch: TensorDict) -> torch.Tensor:
        x = batch["eeg", "data"]
        mask = batch["eeg", "mask"]
        # Flatten all token-like dimensions, keep feature dim D
        b, *token_dims, d = x.shape
        x = x.reshape(b, -1, d)  # [B, N, D]

        mask = mask.reshape(b, -1).bool()  # [B, N]

        # Masked mean pooling over valid tokens
        mask_f = mask.unsqueeze(-1).to(x.dtype)  # [B, N, 1]
        summed = (x * mask_f).sum(dim=1)  # [B, D]
        counts = mask_f.sum(dim=1).clamp_min(1.0)  # [B, 1]
        pooled = summed / counts  # [B, D]

        return self.project(pooled)  # [B, out_dim]


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
        with torch.no_grad():
            y = self.backbone(x)

        # Restore previous batch
        y = y.cls.unflatten(0, (-1, b_inner))
        # AVG over N timesteps of a sample
        y = y.mean(dim=-2)
        return self.project(y)


class SimpleNonLinearProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int, hidden_dim: int = 64):
        super(SimpleNonLinearProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone
        self.hidden = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), )
        self.project = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: TensorDict) -> torch.Tensor:
        # Data inputs are of the shape [B, B', T, P, D]
        b_inner = x.shape[1]
        x = x.flatten(0, 1)
        # Frozen encoder
        with torch.no_grad():
            y = self.backbone(x)

        # Restore previous batch
        y = y.cls.unflatten(0, (-1, b_inner))
        # AVG over N timesteps of a sample
        y = y.mean(dim=-2)

        y = self.hidden(y)
        return self.project(y)
