from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractAttentionBlock(nn.Module, ABC):
    @abstractmethod
    def forward(self, q: torch.Tensor, kv: torch.Tensor, attn_mask=None, q_mask=None, kv_mask=None):
        pass
