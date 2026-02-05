from __future__ import annotations

import torch
from torch import nn as nn, nn
from torch.nn.functional import softmax


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


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dimension: int) -> None:
        """
        Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
        https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
        It gives each token of the input an attention weight for relevance.
        TODO: Not used

        :param input_dimension: Hidden size
        """
        super().__init__()
        self.W = nn.Linear(input_dimension, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = softmax(self.W(x).squeeze(-1)).unsqueeze(-1)
        return torch.sum(x * attn, dim=1)


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
