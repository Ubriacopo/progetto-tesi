from __future__ import annotations

import torch
from torch import nn


class MaskedAttnPool(nn.Module):
    def __init__(self, d, hidden=None):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(d, hidden or d), nn.GELU(), nn.Linear(hidden or d, 1))

    def forward(self, z, mask, dim):
        # z: [..., N, D], mask: [..., N] or None
        w = self.scorer(z).squeeze(-1)  # [..., N]
        if mask is None:
            a = w.softmax(dim=dim).unsqueeze(-1)  # [..., N, 1]
            return (z * a).sum(dim=dim)

        m = mask.to(torch.bool)
        while m.dim() < w.dim():
            m = m.unsqueeze(-1)  # broadcast to w
        m = m.expand_as(w)  # [..., N]

        # mask invalid positions
        w = w.masked_fill(~m, float('-inf'))

        # handle rows with all-masked (would give NaNs)
        all_masked = (~m).all(dim=dim, keepdim=True)  # [..., 1]
        w = torch.where(all_masked.expand_as(w), torch.zeros_like(w), w)

        a = w.softmax(dim=dim)  # [..., N]
        a = a * m  # zero-out invalids
        denom = a.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        a = (a / denom).unsqueeze(-1)  # [..., N, 1]
        return (z * a).sum(dim=dim)


class MaskedMaxPooling(nn.Module):
    # todo review
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # z:    [B, T, D]
        # mask: [B, T]  (1/True = keep, 0/False = ignore)
        mask = mask.detach().unsqueeze(-1).bool()  # [B, T, 1]

        # Set masked positions to -inf so they don't win the max
        z_masked = z.masked_fill(~mask, float("-inf"))  # [B, T, D]
        out = z_masked.max(dim=1).values  # [B, D]

        # If a sequence is fully masked, max is -inf -> replace with zeros
        all_masked = (~mask).all(dim=1).squeeze(-1)  # [B]
        if all_masked.any():
            out = out.masked_fill(all_masked.unsqueeze(-1), 0.0)

        return out


class MaskedAvgPooling(nn.Module):
    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.detach().unsqueeze(-1).float()
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (z * mask).sum(dim=1) / denominator


class ClsPooling(nn.Module):
    def __init__(self, cls_idx: int = -1):
        super().__init__()
        self.cls_idx: int = cls_idx

    def forward(self, z: torch.Tensor, mask: torch.Tensor):
        return z[:, self.cls_idx]
