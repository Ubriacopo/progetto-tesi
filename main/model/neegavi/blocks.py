from abc import ABC, abstractmethod

import torch
from einops import rearrange, repeat
from torch import nn

from main.model.layer.base import ModalContextEncoder
from main.model.layer.kd import KDHead
from main.model.layer.modality_stream import ModalityStream
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


# todo implementa mancante roba e fai per ogni modality uno custom
class MaskedModalityStream(nn.Module, ABC):
    KD_KEY: str = "kd"

    def __init__(self, modal_context_encoder: ModalContextEncoder, code: str, time_step_length: int,
                 kd_head: KDHead = None, post_kd_adapter: nn.Module = None, ):
        super().__init__()
        self.code = code

        self.modal_context_encoder = modal_context_encoder
        self.time_step_length = time_step_length

    @abstractmethod
    def adapt_modality(self, data: torch.Tensor, mask: torch.Tensor = None, use_kd: bool = False) \
            -> MaskedValue | KdMaskedValue:
        pass

    def forward(self, x: MaskedValue, b: int, use_kd: bool, idx: torch.Tensor = None):
        output = {}

        data = x["data"]
        t = data.shape[1]
        mask = x.get("mask", None)

        y: MaskedValue | KdMaskedValue = self.adapt_modality(data, mask=mask, use_kd=use_kd)
        if self.KD_KEY in y:
            output[self.KD_KEY] = y.pop(self.KD_KEY)
            # TODO Play with idx

        z = y["data"]  # We removed KD if it was present
        if self.modal_context_encoder is not None:
            z = self.modal_context_encoder(y["data"], modality=self.code)

        times = torch.arange(t, device=data.device) * self.time_step_length
        m = z.shape[2]

        return output | {
            "data": rearrange(z, "b t m d -> b (t m) d"),
            "mask": repeat(mask, "b t -> b (t m)", m=m),
            "t_mod": repeat(times, "t -> b (t m)", b=b, m=m)
        }
