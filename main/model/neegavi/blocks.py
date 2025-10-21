from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange, repeat
from torch import nn

from main.model.layer.base import ModalContextEncoder
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
                 modal_context_encoder: ModalContextEncoder = None,
                 kd_head: KDHead = None, post_kd_adapter: nn.Module = None):
        super().__init__()

        self.output_size: int = output_size
        self.code: str = code
        self.adapter: nn.Module = adapter

        self.modal_context_encoder: Optional[ModalContextEncoder] = modal_context_encoder
        self.post_kd_adapter: Optional[nn.Module] = post_kd_adapter
        if self.post_kd_adapter is not None and not self.use_kd:
            raise ValueError("You have to use KD to use the post_kd_adapter")

        self.use_kd: bool = kd_head is not None
        self.kd_head: Optional[KDHead] = kd_head

    def init_modal_context_encoder(self, modal_context_encoder: ModalContextEncoder):
        if self.modal_context_encoder is not None:
            raise ValueError("Modal context encoder already initialized")
        self.modal_context_encoder = modal_context_encoder

    def forward(self, x: torch.Tensor, mask=None, use_kd=True, **kwargs) -> MaskedValue | KdMaskedValue:
        output = {"data": x, "mask": mask}
        y: MaskedValue = self.adapter(x, mask=mask)
        if use_kd and self.use_kd:
            output["kd"] = self.kd_head(y["data"], mask=y["mask"])
        if self.post_kd_adapter is not None:
            y |= self.post_kd_adapter()
        if self.modal_context_encoder is not None:
            y["data"] = self.modal_context_encoder(y, modality=self.get_code())
        return output | y

    def get_code(self):
        return self.code

    def as_tuple(self) -> tuple[str, ModalityStream]:
        return self.code, self


class TimedMaskedModalityStream(ModalityStream):
    def __init__(self, code: str, output_size: int, adapter: nn.Module,
                 modal_context_encoder: ModalContextEncoder = None,
                 kd_head: KDHead = None, post_kd_adapter: nn.Module = None,
                 time_step_length: float = 1.):
        super().__init__(code, output_size, adapter, modal_context_encoder, kd_head, post_kd_adapter)
        self.time_step_length: float = time_step_length

    def forward(self, x: torch.Tensor, mask=None, use_kd=True, idx: torch.Tensor = None, b: int = None, **kwargs) \
            -> KdMaskedValue:
        output = {}
        if b is None:
            b = x.shape[0]

        t = x.shape[1]
        y: KdMaskedValue = super().forward(x=x, mask=mask, use_kd=use_kd)

        times = torch.arange(t, device=x.device) * self.time_step_length
        m = y["data"].shape[2]

        return_object = KdMaskedValue(
            data=rearrange(y["data"], "b t m d -> b (t m) d"),
            mask=repeat(mask, "b t -> b (t m)", m=m),
            t_mod=repeat(times, "t -> b (t m)", b=b, m=m),
        )

        if "kd" in y and y["kd"] is not None:
            return_object["kd"] = y["kd"]
            # We removed KD if it was present and move it to output.
            # TODO Play with idx

        return return_object
