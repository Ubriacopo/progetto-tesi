from abc import ABC
from typing import Optional

import torch
from einops import rearrange
from torch import nn

from models.FEEG.layers.base_embedding import FoundationEmbedder
from models.FEEG.layers.kd import KDHead


class EmbeddingsAdapter(nn.Module):
    def __init__(self, embedder: FoundationEmbedder, adapter: nn.Module,
                 target_shape: int, kd_size: int = None):
        """
        Cool to test out different adapting techniques while respecting the required structure by EEGAVI model

        :param embedder:
        :param adapter:
        :param target_shape:
        :param kd_size:
        """
        super().__init__()
        self.adapter = adapter

        self.kd_head = None
        if kd_size is not None:
            self.kd_head = KDHead(input_dimension=embedder.output_size, output_dimension=kd_size)

        self.projection: Optional[nn.Linear] = None
        if target_shape != embedder.output_size:
            self.projection = nn.Linear(embedder.output_size, target_shape)

        self.embedder = embedder

    def forward(self, x, use_kd: bool = True):
        if x is None:
            return x

        z = self.embedder(x)
        z = self.adapter(z)

        kd_z: Optional[torch.Tensor] = None
        if self.kd_head is not None and use_kd:
            kd_z = rearrange(z, "b T n d -> b (T n) d")
            kd_z = kd_z.mean(dim=1)
            kd_z = self.kd_head(kd_z)

        if self.projection is not None:
            z = self.projection(z)

        return (z, kd_z) if use_kd and self.kd_head is not None else (z, None)
