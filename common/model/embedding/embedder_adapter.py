from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from common.model.embedding.foundation_embedder import FoundationEmbedder
from common.model.layers.kd import KDHead


class EmbedderAdapter(nn.Module):
    def __init__(self, embedder: FoundationEmbedder, adapter: nn.Module,
                 output_size: int, kd_shape: Tuple[int, ...] = None):
        """
        Standard structure. If you need to operate after KD explicitly you have to override the class and make a new one.
        For now, I won't as there is no need for the moment.

        :param embedder:
        :param adapter:
        :param kd_shape:
        """
        super(EmbedderAdapter, self).__init__()
        self.embedder: FoundationEmbedder = embedder
        self.adapter: nn.Module = adapter

        self.kd_head = None
        if kd_shape is not None:
            self.kd_head = KDHead(input_size=output_size, target_shape=kd_shape)

    def forward(self, x: torch.Tensor, mask=None, use_kd: bool = True) \
            -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        z = self.embedder(x, mask=mask)
        z = self.adapter(z)

        kd_z: Optional[torch.Tensor] = None
        if use_kd and self.kd_head is not None:
            kd_z = self.kd_head(z)

        return (z, kd_z) if use_kd else z
