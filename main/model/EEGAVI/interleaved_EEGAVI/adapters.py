from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange, repeat
from torch import nn

from main.model.layer.base import TemporalEncoder
from main.model.layer.perceiver_adapter import PerceiverResampler
from main.utils.data import MaskedValue


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


class EegAdapter(nn.Module):
    def __init__(self, channels: int, latent_input_size: int, output_size: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(channels * latent_input_size, output_size),
            nn.GELU(),
            nn.LayerNorm(output_size),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> MaskedValue:
        if mask is not None:
            x *= mask[..., None].to(x.dtype)  # zero masked channels first
        x = rearrange(x, "b T c L -> b T (c L)")
        x = self.ff(x)
        if mask is not None:
            # (b, T) - which time steps have ANY valid channel
            mask = mask.any(dim=-1) if mask is not None else None

        return {"data": x, "mask": mask}


class VideoAdapter(nn.Module):
    def __init__(self, perceiver_config: PerceiverResamplerConfig, project_out_size: int = None, patch_size: int = 49):
        super().__init__()
        self.patch_size = patch_size
        self.resampler = PerceiverResampler(**perceiver_config.__dict__)
        self.projection: Optional[nn.Module] = None
        if project_out_size is not None and project_out_size != perceiver_config.dim:
            self.projection = nn.Linear(perceiver_config.dim, project_out_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = rearrange(x, "b T (F p) D -> b T F p D", p=self.patch_size)
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
    def __init__(self, p: int, perceiver_config: PerceiverResamplerConfig, project_out_size: int = None, ):
        super().__init__()
        self.p: int = p
        self.resampler = TemporalEncoder(dim=384)
        self.projection: Optional[nn.Module] = None
        if project_out_size is not None and project_out_size != perceiver_config.dim:
            self.projection = nn.Linear(perceiver_config.dim, project_out_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        y = self.resampler(x=x, mask=mask)
        # TODO This after KD
        y = repeat(y, "b T D -> b T p D", p=64)
        if self.projection is not None:
            y = self.projection(y)

        return y
