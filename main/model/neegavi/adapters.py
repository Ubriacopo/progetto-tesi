import logging
from dataclasses import dataclass
from typing import Optional

import torch
from einops import repeat, rearrange
from torch import nn

from main.model.neegavi.blocks import TemporalEncoder
from main.model.neegavi.perceiver import PerceiverResampler
from main.utils.data import MaskedValue


@dataclass
class PerceiverResamplerConfig:
    dim: int
    depth: int
    dim_head: int = 64
    heads: int = 8
    num_latents: int = 64
    max_num_time_steps: int = None
    ff_mult: int = 4


class TimedMaskedAdapter(nn.Module):
    pass


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
            x = x * mask[..., None].to(x.dtype)  # zero masked channels first
        x = rearrange(x, "b T c L -> b T (c L)")
        x = self.ff(x)
        if mask is not None:
            # (b, T) - which time steps have ANY valid channel
            mask = mask.any(dim=-1) if mask is not None else None

        return {"data": x, "mask": mask}


class PerceiverResamplerAdapter(nn.Module):
    def __init__(self, perceiver_config: PerceiverResamplerConfig,
                 project_out_size: int = None, post_resample_module: nn.Module = None):
        super().__init__()
        self.resampler = PerceiverResampler(**perceiver_config.__dict__)
        self.post_resample_module: Optional[nn.Module] = post_resample_module
        # We have to adapt
        if self.post_resample_module is None and project_out_size is not None and project_out_size != perceiver_config.dim:
            logging.info(f"Shapes do not match so a nn.Linear({perceiver_config.dim}, {project_out_size}) is created")
            self.post_resample_module = nn.Linear(perceiver_config.dim, project_out_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> MaskedValue:
        y = self.resampler(x=x, mask=mask)
        if self.post_resample_module is not None:
            y = self.post_resample_module(y)

        return MaskedValue(data=y, mask=mask)


class SimpleFeedForwardAdapter(nn.Module):
    def __init__(self, input_size: int, project_out_size: int = None, mult: int = 4):
        assert mult > 0, "Mult has to be positive"
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(input_size, input_size * mult),
            nn.GELU(),
            nn.LayerNorm(input_size * mult),
            nn.Linear(input_size * mult, project_out_size),
        )

    def forward(self, x: torch.Tensor, mask=None):
        """
        :param x: [b T P D]
        :param mask: [b T]
        :return:
        """
        y = self.ff(x)
        return MaskedValue(data=y, mask=mask)


class TemporalEncoderAdapter(nn.Module):
    def __init__(self, p: int, dim: int, project_out_size: int = None):
        super().__init__()
        self.p: int = p
        self.temporal_encoder = TemporalEncoder(dim=dim)
        self.projection: Optional[nn.Module] = None
        if project_out_size is not None and project_out_size != dim:
            self.projection = nn.Linear(dim, project_out_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> MaskedValue:
        # TODO Verifica
        y = self.temporal_encoder(x=x, mask=mask)
        y = repeat(y, "b T D -> b T p D", p=self.p)

        if self.projection is not None:
            y = self.projection(y)

        return MaskedValue(data=y, mask=mask)
