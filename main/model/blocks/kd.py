from typing import Optional

import torch
import torch.nn as nn

from main.utils.data import MaskedValue


class KDHead(nn.Module):
    def __init__(self, input_size: int, target_size: int, return_masks: bool = True):
        """
        Has the task to project to same dimension and shape for KD loss computation.
        Inputs in my model are mostly 4D while the teacher is 1D so we have to pool dimensions.
        """
        super().__init__()
        # KISS (Keep it Simple Stupid)
        self.transform = nn.Sequential(
            nn.Linear(input_size, target_size),
            nn.LayerNorm(target_size),
        )

        self.return_masks = return_masks
        self.eps = 1e-8

    @staticmethod
    def masked_mean(x, mask, dim, eps=1e-8):
        if mask is None:
            pooled = x.mean(dim=dim)
            valid = torch.ones_like(pooled[..., :1], dtype=torch.bool, device=x.device)

            return pooled, valid

        m = mask.to(dtype=x.dtype)
        while m.dim() < x.dim():
            m = m.unsqueeze(-1)  # [...,N,1]

        num = (x * m).sum(dim=dim)
        den = m.sum(dim=dim)

        valid = den > 0
        pooled = num / den.clamp_min(eps)
        pooled = torch.where(valid, pooled, torch.zeros_like(pooled))

        return pooled, valid

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: [B,T,P,IN], mask: [B,T,P] or None
        zTP, valid_T = self.masked_mean(x, mask, dim=2, eps=self.eps)  # [B,T,IN], [B,T,1]
        mT = valid_T.squeeze(-1)  # [B,T] bool
        zB, valid_B = self.masked_mean(zTP, mT, dim=1, eps=self.eps)  # [B,IN], [B,1]
        y = self.transform(zB)  # [B,OUT]
        out_mask = valid_B.squeeze(-1)

        return y if not self.return_masks else MaskedValue(data=y, mask=out_mask)
