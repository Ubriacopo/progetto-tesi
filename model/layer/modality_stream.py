from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from model.layer.kd import KDHead
from utils.data import MaskedValue


class ModalityStream(nn.Module):
    def __init__(self, code: str, adapter: nn.Module, kd_head: KDHead = None, post_kd_adapter: nn.Module = None,
                 drop_mask: bool = False):
        super().__init__()
        self.code: str = code
        self.adapter: nn.Module = adapter

        self.post_kd_adapter: Optional[nn.Module] = post_kd_adapter
        if self.post_kd_adapter is not None and not self.use_kd:
            raise ValueError("You have to use KD to use the post_kd_adapter")

        self.use_kd: bool = kd_head is not None
        self.drop_mask: bool = drop_mask
        self.kd_head: Optional[KDHead] = kd_head

    def forward(self, x: torch.Tensor, mask=None, use_kd=True) \
            -> MaskedValue | tuple[MaskedValue, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        y = self.adapter(x, mask=mask)
        if isinstance(y, tuple) and len(y) == 2:
            y, mask = y
        elif isinstance(y, dict):
            y, mask = y["data"], y["mask"]

        use_kd = use_kd and self.use_kd
        kd_y: Optional[torch.Tensor] = None
        if use_kd:
            kd_y = self.kd_head(y, mask=mask)

        if self.post_kd_adapter is not None:
            y = self.post_kd_adapter(y)

        if mask is not None and not self.drop_mask:
            y = MaskedValue(data=y, mask=mask)
        return (y, kd_y) if use_kd else y

    def get_code(self):
        return self.code

    def as_tuple(self) -> tuple[str, ModalityStream]:
        return self.code, self
