from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from main.model.layer.kd import KDHead
from main.utils.data import MaskedValue, KdMaskedValue


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


class ModalityStream(nn.Module):
    def __init__(self, code: str, output_size: int, adapter: nn.Module,
                 kd_head: KDHead = None, post_kd_adapter: nn.Module = None, time_step_length: float = 1.0):
        super().__init__()

        self.output_size: int = output_size
        self.code: str = code
        self.adapter: nn.Module = adapter

        self.post_kd_adapter: Optional[nn.Module] = post_kd_adapter
        if self.post_kd_adapter is not None and not self.use_kd:
            raise ValueError("You have to use KD to use the post_kd_adapter")

        self.use_kd: bool = kd_head is not None
        self.kd_head: Optional[KDHead] = kd_head
        self.time_step_length: float = time_step_length

    def forward(self, x: torch.Tensor, mask=None, use_kd=True, **kwargs) -> MaskedValue | KdMaskedValue:
        output = {"data": x, "mask": mask}
        y: MaskedValue = self.adapter(x, mask=mask)
        if use_kd and self.use_kd:
            output["kd"] = self.kd_head(y["data"], mask=y["mask"])
        if self.post_kd_adapter is not None:
            y |= self.post_kd_adapter()
        return output | y

    def get_code(self):
        return self.code

    def as_tuple(self) -> tuple[str, ModalityStream]:
        return self.code, self
