from typing import Optional

import torch
from einops import rearrange
from torch import nn

from common.model.layers.ISAB import ISAB, PMA


class ISABPMATimeAdapter(nn.Module):
    def __init__(self, output_size: int,
                 isab_enabled: bool = True,
                 isab_num_heads: int = 8, isab_num_i: int = 10,
                 pma_num_heads: int = 8, pma_num_i: int = 10, ):
        super().__init__()

        self.isab: Optional[ISAB] = None
        if isab_enabled:
            self.isab = ISAB(output_size, isab_num_heads, isab_num_i)

        self.pma = PMA(output_size, pma_num_heads, pma_num_i)

    def forward(self, z: torch.Tensor):
        b = z.shape[0]  # Batch size is leftmost

        z = rearrange(z, 'b T F D -> (b T) F D')
        if self.isab is not None:
            z = self.isab(z)
        z = self.pma(z)

        z = rearrange(z, '(b T) F D -> b T F D', b=b)
        return z
