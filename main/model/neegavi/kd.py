from typing import Tuple, Optional

import torch
from einops import repeat
from torch import nn

from main.utils.data import MaskedValue

import torch, torch.nn as nn, torch.nn.functional as F


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


class AKDHead(nn.Module):
    def __init__(self, input_size, target_shape, return_masks=True):
        super().__init__()
        D = 384
        self.poolP = MaskedAttnPool(D, hidden=D)  # or MaskedGeM()
        self.poolT = MaskedAttnPool(D, hidden=D)

        self.transform = nn.Sequential(
            nn.Linear(input_size, 4 * 100), nn.GELU(),
            nn.LayerNorm(4 * 100),
            nn.Linear(4 * 100, 100), nn.LayerNorm(100)
        )
        self.return_masks = return_masks
        self.target_shape = target_shape

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: [B,T,P,IN], mask: [B,T,P] or None
        mP = repeat(mask, "b t -> b t p", p=x.shape[-2])
        zTP = self.poolP(x, mP, dim=2)  # [B,T,IN]
        mT = None if mP is None else mP.any(dim=2)  # [B,T] bool
        zB = self.poolT(zTP, mT, dim=1)  # [B,IN]
        y = self.transform(zB)  # [B,OUT]
        out_mask = torch.ones(y.size(0), dtype=torch.bool, device=y.device) if mT is None else mT.any(dim=1)
        return y if not self.return_masks else MaskedValue(data=y, mask=out_mask)


class KDHead(nn.Module):
    def __init__(self, input_size, target_shape, return_masks: bool = True):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(input_size, 4 * target_shape[-1]),
            nn.GELU(),
            nn.LayerNorm(4 * target_shape[-1]),
            nn.Linear(4 * target_shape[-1], target_shape[-1]),
            nn.LayerNorm(target_shape[-1])
        )
        self.return_masks = return_masks
        self.eps = 1e-8

    @staticmethod
    def masked_mean(x, mask, dim, eps=1e-8):
        if mask is None:
            pooled = x.mean(dim=dim)
            valid = torch.ones_like(pooled[..., :1], dtype=torch.bool, device=x.device)
            return pooled, valid
        m = mask.float()
        while m.dim() < x.dim(): m = m.unsqueeze(-1)  # [...,N,1]
        num = (x * m).sum(dim=dim)
        den = m.sum(dim=dim).clamp_min(1.0)
        pooled = num / den
        valid = (den > 0).to(torch.bool)
        return pooled, valid

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: [B,T,P,IN], mask: [B,T,P] or None
        zTP, valid_T = self.masked_mean(x, mask, dim=2)  # [B,T,IN], [B,T,1]
        mT = None if valid_T is None else valid_T.squeeze(-1)  # [B,T] bool
        zB, valid_B = self.masked_mean(zTP, mT, dim=1)  # [B,IN], [B,1]
        y = self.transform(zB)  # [B,OUT]
        out_mask = valid_B.squeeze(-1) if valid_B is not None else torch.ones(y.size(0), dtype=torch.bool,
                                                                              device=y.device)
        return y if not self.return_masks else MaskedValue(data=y, mask=out_mask)
