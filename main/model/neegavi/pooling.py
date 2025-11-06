from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import softmax


class MaskedPooling(nn.Module):
    """
    Simple fusion pooling on mask
    """

    # noinspection PyMethodMayBeStatic
    def forward(self, z: torch.Tensor, mask=None) -> torch.Tensor:
        norm_factor = mask.float().sum(dim=-1, keepdim=True)
        norm_factor = norm_factor.clamp_min(1e-6)
        z = (z * mask.unsqueeze(-1)).sum(dim=-2) / norm_factor

        return z

# todo volevo usarlo?
class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dimension: int) -> None:
        """
        Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
        https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
        It gives each token of the input an attention weight for relevance.

        :param input_dimension: Hidden size
        """
        super().__init__()
        self.W = nn.Linear(input_dimension, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = softmax(self.W(x).squeeze(-1)).unsqueeze(-1)
        return torch.sum(x * attn, dim=1)
