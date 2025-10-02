from typing import Tuple, Optional

import torch
from torch import nn


class KDHead(nn.Module):
    def __init__(self, input_size: int, target_shape: Tuple[int, ...], transform: nn.Module = None,
                 normalize: bool = True):
        super(KDHead, self).__init__()
        self.transform = nn.Linear(input_size, target_shape[-1]) if transform is None else transform
        self.target_shape = target_shape
        self.normalize: bool = normalize
        self.eps = 1e-9

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int, eps: float):
        if mask is None:
            return x.mean(dim=dim)

        mask = mask.to(dtype=x.dtype)
        return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp_min(eps)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 4 and len(self.target_shape) <= 3:
            mask4d = None
            if mask is not None:
                mask4d = mask[:, :, None] if mask.dim() == 2 else mask
                assert mask4d.dim() == 3, "mask must be (B,T) or (B,T,P)"
                mask4d = mask4d[:, :, :, None]  # (B, T, P, 1)

            x = self.masked_mean(x, mask4d, dim=-2, eps=self.eps)  # → (B,T,D)

        if x.dim() == 3 and len(self.target_shape) == 2:
            mask3d = None
            if mask is not None:
                mask3d = mask.any(dim=-1) if mask.dim() == 3 else mask
                assert mask3d.dim() == 2, "mask must be (B,T) or (B,T,P)"
                mask3d = mask3d[:, :, None]  # (B, T, 1)

            x = self.masked_mean(x, mask3d, dim=-2, eps=self.eps)  # → (B,T,D)

        if x.dim() != len(self.target_shape):
            raise ValueError(f"Shape mismatch after pooling: x:{tuple(x.shape)} vs target:{self.target_shape}")

        y = self.transform(x)
        # L2-normalization
        return torch.nn.functional.normalize(y, p=2, dim=-1, eps=self.eps) if self.normalize else y
