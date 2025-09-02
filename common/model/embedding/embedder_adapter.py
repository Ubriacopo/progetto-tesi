from typing import Optional

import torch
from einops import rearrange
from torch import nn

from common.model.embedding.foundation_embedder import FoundationEmbedder
from common.model.layers.kd import KDHead


class EmbedderAdapter(nn.Module):
    def __init__(self, embedder: FoundationEmbedder, adapter: nn.Module, target_size: int, kd_size: int = None):
        """
        Cool to test out different adapting techniques while respecting the required structure by EEGAVI model

        :param embedder: The embedding model to be called.
        :param adapter: The adapter model to be called.
        :param target_size: The target size of the embedding. If it is the same as the one of the embedder it won't be generated.
        :param kd_size: Size for KD. If none KD is disabled for this adapter.
        """
        super().__init__()
        # Callable adapter model
        self.adapter: nn.Module = adapter
        self.embedder: FoundationEmbedder = embedder

        self.kd_head = None
        if kd_size is not None:
            self.kd_head = KDHead(input_dimension=embedder.output_size, output_dimension=kd_size)

        self.projection: Optional[nn.Linear] = None
        if target_size != embedder.output_size:
            self.projection = nn.Linear(embedder.output_size, target_size)

    def forward(self, x, mask=None, use_kd: bool = True):
        # TODO: masking al posto di none?
        if x is None:
            return x

        z = self.embedder(x)
        z = self.adapter(z)

        kd_z: Optional[torch.Tensor] = None
        if self.kd_head is not None and use_kd:
            kd_z = z
            # If the input uses time series
            if len(z.shape) == 4:
                kd_z = rearrange(kd_z, "b T n d -> b (T n) d")

            kd_z = self.kd_head(kd_z.mean(dim=1))

        if self.projection is not None:
            z = self.projection(z)

        return (z, kd_z) if use_kd and self.kd_head is not None else z
