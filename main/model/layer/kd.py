from typing import Tuple, Optional

import torch
from torch import nn


class KDHead(nn.Module):
    def __init__(self, input_size: int, target_shape: Tuple[int, ...], transform: nn.Module = None,
                 normalize: bool = False, return_masks: bool = True):
        super(KDHead, self).__init__()
        self.transform = nn.Linear(input_size, target_shape[-1]) if transform is None else transform
        self.target_shape = target_shape
        self.normalize: bool = normalize
        self.return_masks: bool = return_masks
        self.eps = 1e-9

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float):
        if mask is None:
            pooled = x.mean(dim=dim)
            valid = torch.ones_like(pooled.select(-1, 0), dtype=torch.bool)
            return pooled, valid

        mask = mask.to(dtype=x.dtype)
        mask_sum = mask.sum(dim=dim, keepdim=True)

        valid = mask_sum > 0

        numerator = (x * mask).sum(dim=dim)
        denominator = mask_sum.squeeze(dim).clamp_min(eps)

        pooled = torch.where(valid.squeeze(dim), numerator / denominator, torch.zeros_like(numerator))
        return pooled, valid.squeeze(dim)

    # todo review  dopo modifiche fatte
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor] | torch.Tensor:
        out_mask = mask
        if x.dim() == 4 and len(self.target_shape) <= 3:
            mask4d = None
            if mask is not None:
                mask4d = mask[:, :, None] if mask.dim() == 2 else mask
                assert mask4d.dim() == 3, "mask must be (B,T) or (B,T,P)"
                mask4d = mask4d[:, :, :, None]  # (B, T, P, 1)

            x, valid_tp = self.masked_mean(x, mask4d, dim=-2, eps=self.eps)  # (B,T,D), valid_tp: (B,T,1)
            out_mask = valid_tp

        if x.dim() == 3 and len(self.target_shape) == 2:
            mask3d = None
            if mask is not None:
                mask3d = mask.any(dim=-1) if mask.dim() == 3 else mask
                assert mask3d.dim() == 2, "mask must be (B,T) or (B,T,P)"
                mask3d = mask3d[:, :, None]  # (B, T, 1)

            # prefer the already reduced mask if we have it
            if out_mask is not None:
                mask3d = out_mask[:, :, None].any(dim=-1)  # (B,T,1)

            x, valid_t = self.masked_mean(x, mask3d, dim=-2, eps=self.eps)  # (B,D), valid_t: (B,1)
            out_mask = valid_t.squeeze(-1)

        if x.dim() != len(self.target_shape):
            raise ValueError(f"Shape mismatch after pooling: x:{tuple(x.shape)} vs target:{self.target_shape}")

        y = self.transform(x)
        # L2-normalization
        if self.normalize:
            y = torch.nn.functional.normalize(y, p=2, dim=-1, eps=self.eps)

        if y.dim() != len(self.target_shape):
            y = y.squeeze(dim=-1)
        if out_mask is None:
            raise ValueError("Somehow you got no mask")

        return y if not self.return_masks else {"data": y, "mask": out_mask}
