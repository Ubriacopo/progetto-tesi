from typing import Tuple, Optional

import torch
from einops import repeat
from torch import nn

from main.utils.data import MaskedValue

import torch, torch.nn as nn, torch.nn.functional as F


class KDHead(nn.Module):
    def __init__(self, input_size: int, target_shape: Tuple[int, ...],
                 transform: nn.Module = None, return_masks: bool = True):
        super(KDHead, self).__init__()

        self.transform = nn.Linear(input_size, target_shape[-1]) if transform is None else transform

        self.target_shape = target_shape
        self.return_masks: bool = return_masks
        self.eps = 1e-9

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float):
        if mask is None:
            return x.mean(dim=dim), torch.ones(x.shape[:dim] + x.shape[dim + 1:-1], dtype=torch.bool, device=x.device)

        mask = mask.to(dtype=x.dtype)
        while mask.dim() < x.dim():
            mask = mask.unsqueeze(-1)

        mask_sum = mask.sum(dim=dim, keepdim=True)
        valid = mask_sum > 0

        numerator = (x * mask).sum(dim=dim)
        denominator = mask_sum.squeeze(dim).clamp_min(eps)

        pooled = torch.where(valid.squeeze(dim), numerator / denominator, torch.zeros_like(numerator))
        return pooled, valid.squeeze(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> MaskedValue | torch.Tensor:
        out_mask = None
        if x.dim() == 4 and len(self.target_shape) <= 3:
            mP = None if mask is None else (mask if mask.dim() == 3 else mask.unsqueeze(-1))
            x, valid_tp = self.masked_mean(x, mP, dim=2, eps=self.eps)  # (B,T,D), valid_tp: (B,T,1)
            out_mask = valid_tp.squeeze(-1)

        if x.dim() == 3 and len(self.target_shape) == 2:
            mT = out_mask if out_mask is not None else (
                None if mask is None else (mask if mask.dim() == 2 else mask.any(dim=-1)))
            x, valid_t = self.masked_mean(x, mT, dim=1, eps=self.eps)  # (B,D), valid_t: (B,1)
            out_mask = valid_t.squeeze(-1)

        if x.dim() != len(self.target_shape):
            raise ValueError(f"Shape mismatch after pooling: x:{tuple(x.shape)} vs target:{self.target_shape}")

        y = self.transform(x)
        if out_mask is None:
            raise ValueError("Somehow you got no mask")

        return y if not self.return_masks else MaskedValue(data=y, mask=out_mask)



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
