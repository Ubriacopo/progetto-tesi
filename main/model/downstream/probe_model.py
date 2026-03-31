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
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

    def forward(self, x: TensorDict) -> torch.Tensor:
        # Data inputs are of the shape [B, B', T, P, D]
        b, b_inner = x.shape[:2]
        x = x.flatten(0, 1)
        # Frozen encoder
        with torch.no_grad():
            y = self.backbone(x)

        # Restore previous batch
        y = y.cls.unflatten(0, (b, b_inner))
        # AVG over N timesteps of a sample
        # y = y.max(dim=-2).values
        y = y.mean(dim=-2)
        logits = self.project(y)
        return logits


class PooledLinearProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int):
        super(PooledLinearProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone
        self.project = nn.Linear(in_dim * 2, out_dim)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

    def forward(self, x: TensorDict) -> torch.Tensor:
        # Data inputs are of the shape [B, B', T, P, D]
        b, b_inner = x.shape[:2]
        x = x.flatten(0, 1)
        # Frozen encoder
        with torch.no_grad():
            y = self.backbone(x)

        # Restore previous batch
        # [B, B1, 32, 384] (Drop CLS token)
        z = y.embeddings["data"].unflatten(0, (b, b_inner))[:, :, :-1]
        # [B, B1, 32] (Drop CLS Token)
        mask = y.embeddings["mask"].unflatten(0, (b, b_inner))[:, :, :-1]

        mask_f = mask.to(z.dtype)

        # MEAN
        tok_mean = (z * mask_f.unsqueeze(-1)).sum(dim=2) / mask_f.sum(dim=2, keepdim=True).clamp_min(1.0)
        # MAX
        # token-level masked max -> [B, B', D]
        neg_inf = torch.finfo(z.dtype).min
        tok_max = z.masked_fill(~mask.unsqueeze(-1), neg_inf).max(dim=2).values

        # chunk-level mean/max -> [B, D]
        sample_mean = tok_mean.mean(dim=1)
        sample_max = tok_max.max(dim=1).values

        sample_emb = torch.cat([sample_mean, sample_max], dim=-1)  # [B, 2D]
        # AVG over N timesteps of a sample

        logits = self.project(sample_emb)
        return logits


class SimpleNonLinearProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super(SimpleNonLinearProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone
        self.hidden = nn.Sequential(nn.Linear(in_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(0.1), )
        self.project = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: TensorDict) -> torch.Tensor:
        # Data inputs are of the shape [B, B', T, P, D]
        b, b_inner = x.shape[:2]
        x = x.flatten(0, 1)
        # Frozen encoder
        with torch.no_grad():
            y = self.backbone(x)

        # Restore previous batch
        # [B, B1, 32, 384] (Drop CLS token)
        z = y.embeddings["data"].unflatten(0, (b, b_inner))[:, :, :-1]
        # [B, B1, 32] (Drop CLS Token)
        mask = y.embeddings["mask"].unflatten(0, (b, b_inner))[:, :, :-1]

        mask_f = mask.to(z.dtype)

        # MEAN
        tok_mean = (z * mask_f.unsqueeze(-1)).sum(dim=2) / mask_f.sum(dim=2, keepdim=True).clamp_min(1.0)
        # MAX
        # token-level masked max -> [B, B', D]
        neg_inf = torch.finfo(z.dtype).min
        tok_max = z.masked_fill(~mask.unsqueeze(-1), neg_inf).max(dim=2).values

        # chunk-level mean/max -> [B, D]
        sample_mean = tok_mean.mean(dim=1)
        sample_max = tok_max.max(dim=1).values

        sample_emb = torch.cat([sample_mean, sample_max], dim=-1)  # [B, 2D]
        # AVG over N timesteps of a sample
        y = self.hidden(sample_emb)
        return self.project(y)
