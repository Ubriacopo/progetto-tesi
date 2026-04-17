import torch
from cbramod.models.cbramod import CBraMod
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tensordict import TensorDict
from torch import nn

from main.model.neegavi.adapters import EegCbraModAdapter
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


class Simple1DLinearProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int):
        super(Simple1DLinearProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone
        self.project = nn.Sequential(
            nn.Linear(in_dim, out_dim),
        )

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

    def forward(self, x: TensorDict) -> torch.Tensor:
        # Data inputs are of the shape [B, B1, T, P, D]
        # Frozen encoder
        with torch.no_grad():
            y = self.backbone(x)

        # Restore previous batch
        y = y.cls
        # AVG over N timesteps of a sample
        # y = y.max(dim=-2).values
        logits = self.project(y)
        return logits


class Simple1ZFF(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int):
        super(Simple1ZFF, self).__init__()
        self.backbone: EegInterAviModel = backbone
        self.project = nn.Sequential(
            nn.Linear(32 * in_dim, 1024),
            # nn.Linear(in_dim, 32),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(1024, out_dim)
        )
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

    def forward(self, x: TensorDict) -> torch.Tensor:
        # Data inputs are of the shape [B, B1, T, P, D]
        # Frozen encoder
        with torch.no_grad():
            y = self.backbone(x)

        # Restore previous batch
        mask = y.embeddings["mask"][:, :32]
        y = y.embeddings["data"][:, :32]
        D = y.shape[-1]

        z = rearrange(y, "b P D -> b (P D)")
        mask = repeat(mask, "b P -> b (P D)", D=D)

        z = z.masked_fill(~mask, 0.0)  # or z = z * mask if numeric mask
        # AVG over N timesteps of a sample
        # y = y.max(dim=-2).values
        logits = self.project(z)
        return logits


class SimpleLinearProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int):
        super(SimpleLinearProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone
        #  self.project = nn.Linear(in_dim, out_dim)
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
        z = y.cls.unflatten(0, (b, b_inner))
        # z = z.flatten(-2, -1)
        z = z.mean(dim=1)
        logits = self.project(z)
        return logits


class SimplePoolingProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int):
        super(SimplePoolingProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone
        #  self.project = nn.Linear(in_dim, out_dim)
        self.project = nn.Sequential(
            Rearrange('b t d -> b d t'),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_dim, out_dim),
        )

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
        z = y.cls.unflatten(0, (b, b_inner))
        logits = self.project(z)
        return logits


class SimpleExpandedProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int, k: int = 3):
        super(SimpleExpandedProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone
        #  self.project = nn.Linear(in_dim, out_dim)
        self.project = nn.Sequential(
            nn.Linear(in_dim * k, out_dim),
        )

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
        z = y.cls.unflatten(0, (b, b_inner))
        z = z.flatten(-2, -1)
        logits = self.project(z)
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


class SimpleFineTuneProbe(nn.Module):
    def __init__(self, backbone: EegInterAviModel, in_dim: int, out_dim: int, hidden_dim: int = 128):
        super(SimpleFineTuneProbe, self).__init__()
        self.backbone: EegInterAviModel = backbone

        # Freeze CBraMod except last layers
        m: EegCbraModAdapter = self.backbone.pivot.adapter
        for p in m.encoder.parameters():
            p.requires_grad = False

        for p in m.encoder.proj_out.parameters():
            p.requires_grad = True

        for l in m.encoder.encoder.layers[-1:]:
            for p in l.parameters():
                p.requires_grad = True

        self.project = nn.Sequential(
            Rearrange('b t d -> b d t'),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: TensorDict) -> torch.Tensor:
        b, b_inner = x.shape[:2]
        x = x.flatten(0, 1)
        y = self.backbone(x)
        z = y.cls.unflatten(0, (b, b_inner))
        logits = self.project(z)
        return logits


class SimpleCbraFineTune(nn.Module):
    def __init__(self, encoder: CBraMod, in_dim: int, out_dim: int):
        super().__init__()
        self.encoder = encoder

        self.encoder.proj_out = nn.Identity()
        self.project = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, out_dim),
        )

        for p in self.encoder.parameters():
            p.requires_grad = False

        for p in self.encoder.proj_out.parameters():
            p.requires_grad = True

        for l in self.encoder.encoder.layers[-2:]:
            for p in l.parameters():
                p.requires_grad = True

    def forward(self, batch: TensorDict) -> torch.Tensor:
        x = batch["eeg", "data"]  # e.g. [B, B_inner, P, C, D] or similar
        mask = batch["eeg", "mask"]  # same token structure as encoder input, without D

        b, b_inner = x.shape[:2]
        # Flatten outer batch dims for encoder
        x_flat = x.flatten(0, 1)
        mask_flat = mask.flatten(0, 1).bool()

        # Encoder output keeps token structure, only batch is flattened
        y = self.encoder(x_flat.float(), mask_flat)

        # Restore outer batch dims
        y = y.unflatten(0, (b, b_inner))
        mask = mask_flat.unflatten(0, (b, b_inner))
        y = rearrange(y, 'b ... d -> b (...) d')
        mask = rearrange(mask, 'b ... -> b (...)').bool()

        token_logits = self.project(y)  # [B, N, C]

        mask_f = mask.unsqueeze(-1).to(token_logits.dtype)
        logits = (token_logits * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        return logits
