from __future__ import annotations
from typing import Tuple, Optional

import torch
from torch import nn

from model.layer.kd import KDHead


class ModalityStream(nn.Module):
    def __init__(self, code: str, adapter: nn.Module, adapter_output_size: int,
                 # None base as they might just be disabled
                 kd_head: KDHead = None, kd_shape: Tuple[int, ...] = None, post_kd_adapter: nn.Module = None):
        super().__init__()
        self.code: str = code
        self.adapter: nn.Module = adapter

        self.use_kd = kd_head is not None or kd_shape is not None
        self.post_kd_adapter: Optional[nn.Module] = post_kd_adapter
        if self.post_kd_adapter is not None and not self.use_kd:
            raise ValueError("You have to use KD to use the post_kd_adapter")

        self.kd_head: Optional[KDHead] = kd_head
        # Means we use the default KDHead
        if self.kd_head is None and self.use_kd:
            self.kh_head = KDHead(input_size=adapter_output_size, target_shape=kd_shape)

    def forward(self, x: dict | torch.Tensor, mask=None, use_kd=True) -> torch.Tensor:
        # Pretty straightforward todo ragiona masking ora sta in dict quindi ci va anche bene.
        y = self.adapter(x)
        use_kd = use_kd and self.use_kd

        kd_y: Optional[torch.Tensor] = None
        if use_kd:
            kd_y = self.kh_head(y)

        if self.post_kd_adapter is not None:
            y = self.post_kd_adapter(y)

        return (y, kd_y) if use_kd else y

    def get_code(self):
        return self.code

    def as_tuple(self) -> tuple[str, ModalityStream]:
        return self.code, self
