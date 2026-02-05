from __future__ import annotations

import torch
from torch import nn

from main.utils.data import MaskedValue


class MaskedFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float):
        super().__init__()
        self.net = SimpleFeedForward(dim, mult)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask):
        # x [b, T, p, D]
        # m [b, T]
        m = mask[:, :, None, None].to(x.dtype)
        xm = x * m
        y = xm + self.dropout(self.net(xm))
        return MaskedValue(data=y * m, mask=mask)


class SimpleFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int) -> None:
        """
        Two layered feed forward network with GELU activation
        :param dim: Latent space dimension
        :param mult: Feed-forward multiplier
        """
        super().__init__()
        assert mult > 0, "Multiplication has to be a positive integer"
        x, y = dim, dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(x),  # Normalize
            nn.Linear(x, y, bias=False),  # Map to new shape
            nn.GELU(),  # Non-linearity
            nn.Linear(y, x, bias=False),  # Rebuild the original shape
        )

    def forward(self, x):
        return self.net(x)
