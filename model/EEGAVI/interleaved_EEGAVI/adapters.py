from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import nn

from model.layer.ISAB import PMA
from model.layer.base import TemporalEncoder
from model.layer.perceiver_adapter import PerceiverResampler


@dataclass
class PerceiverResamplerConfig:
    dim: int
    depth: int
    dim_head: int = 64
    heads: int = 8
    num_latents: int = 64
    max_num_media: int = None
    max_num_frames: int = None
    ff_mult: int = 4


class VideoAdapter(nn.Module):
    def __init__(self, perceiver_config: PerceiverResamplerConfig, project_out_size: int = None, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.resampler = PerceiverResampler(**perceiver_config.__dict__)
        self.projection: Optional[nn.Module] = None
        if project_out_size is not None and project_out_size != perceiver_config.dim:
            self.projection = nn.Linear(perceiver_config.dim, project_out_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = rearrange(x, "b T (F p) D -> b T F p D", F=self.patch_size)
        y = self.resampler(x=x, mask=mask)
        if self.projection is not None:
            y = self.projection(y)

        return y


class AudioAdapter(nn.Module):
    def __init__(self, perceiver_config: PerceiverResamplerConfig, project_out_size: int = None, ):
        super().__init__()
        self.resampler = PerceiverResampler(**perceiver_config.__dict__)
        self.projection: Optional[nn.Module] = None
        if project_out_size is not None and project_out_size != perceiver_config.dim:
            self.projection = nn.Linear(perceiver_config.dim, project_out_size)

    def forward(self, x: dict, mask=None):
        # Audio embeddings have no further decomposition. We simply add a dim to fit requirements
        x = rearrange(x, "b T (F p) D -> b T F p D", F=1)
        y = self.resampler(x=x, mask=mask)
        if self.projection is not None:
            y = self.projection(y)

        return y


class TextAdapter(nn.Module):
    def __init__(self, perceiver_config: PerceiverResamplerConfig, project_out_size: int = None, ):
        super().__init__()
        self.resampler = TemporalEncoder(dim=384)
        self.projection: Optional[nn.Module] = None
        if project_out_size is not None and project_out_size != perceiver_config.dim:
            self.projection = nn.Linear(perceiver_config.dim, project_out_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Audio embeddings have no further decomposition. We simply add a dim to fit requirements
        y = self.resampler(x=x, mask=mask)
        if self.projection is not None:
            y = self.projection(y)
        return y
