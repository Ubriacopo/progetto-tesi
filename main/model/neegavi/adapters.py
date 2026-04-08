from dataclasses import dataclass, asdict
from typing import Optional

import torch
from cbramod.models.cbramod import CBraMod
from einops import rearrange
from torch import nn

from main.model.blocks.encoder import TemporalEncoder
from main.model.blocks.time_masked import TimeMaskSwitchableProperties
from main.model.blocks.feed_forward import MaskedFeedForward
from main.model.blocks.perceiver import PerceiverResampler
from main.utils.data import MaskedValue
from main.utils.logging import make_logger


@dataclass
class PerceiverResamplerConfig:
    dim: int
    depth: int
    dim_head: int = 64
    heads: int = 12
    num_latents: int = 64
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

        # TODO pool here per time alignment
        return MaskedValue(data=x, mask=mask)


class EegCbraModAdapter(nn.Module):
    def __init__(self, weights_path: str, output_size: int):
        super().__init__()
        self.encoder = CBraMod()
        self.encoder.load_state_dict(torch.load(weights_path))
        self.adapter = EegAdapter(channels=30, latent_input_size=200, output_size=output_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> MaskedValue:
        # Is already time padded.
        # TODO Canonical transform.


        x = x.float()
        if mask is not None:
            mask = mask.bool()
        z = self.encoder(x=x, mask=mask)

        # todo custom adapter part
        z = self.adapter(x=z, mask=mask)
        return z


class PerceiverResamplerAdapter(nn.Module):
    def __init__(self, perceiver_config: PerceiverResamplerConfig, in_size: int,
                 project_out_size: int = None, post_resample_module: nn.Module = None):
        super().__init__()
        self.logger = make_logger(self.__class__.__name__)
        self.linear_reshape: nn.Module = nn.Identity()

        perceiver_config.dim = in_size  # Make sure the dims match
        if project_out_size is not None and project_out_size != in_size:
            # TODO: In case this is noisy just to MLP + LN + non-linearity
            self.logger.info(f"Shapes do not match so a nn.Linear({in_size}, {project_out_size}) is created")
            perceiver_config.dim = project_out_size
            self.linear_reshape = nn.Linear(in_size, project_out_size)

        if isinstance(perceiver_config, PerceiverResamplerConfig):
            perceiver_config = asdict(perceiver_config)

        self.resampler = PerceiverResampler(**perceiver_config)
        self.post_resample_module: Optional[nn.Module] = post_resample_module

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> MaskedValue:
        y = self.linear_reshape(x)
        y = self.resampler(x=y, mask=mask)
        if self.post_resample_module is not None:
            y = self.post_resample_module(y)

        return MaskedValue(data=y, mask=mask)


class SimpleFeedForwardAdapter(nn.Module):
    def __init__(self, in_size: int, project_out_size: int = None, mult: int = 4, dropout: float = .0):
        assert mult > 0, "Mult has to be positive"
        super().__init__()
        self.linear_reshape: nn.Module = nn.Identity()

        out_size = in_size

        if project_out_size is not None and project_out_size != in_size:
            self.linear_reshape = nn.Linear(in_size, project_out_size)
            out_size = project_out_size

        self.masked_ff = MaskedFeedForward(out_size, mult=mult, dropout=dropout)

    def forward(self, x: torch.Tensor, mask=None):
        """
        :param x: [b T P D]
        :param mask: [b T]
        :return:
        """
        return self.masked_ff(self.linear_reshape(x), mask=mask)


class TemporalEncoderAdapter(nn.Module):
    def __init__(self, dim: int, max_length: int, timestep_duration: int,
                 modality: TimeMaskSwitchableProperties, project_out_size: int = None):
        super().__init__()

        self.projection: nn.Module = nn.Identity()
        if project_out_size is not None and project_out_size != dim:
            self.projection = nn.Linear(dim, project_out_size)
            dim = project_out_size

        self.temporal_encoder: nn.Module = TemporalEncoder(
            dim=dim, max_length=max_length, timestep_duration=timestep_duration, modality=modality,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> MaskedValue:
        y = self.projection(x)
        y = self.temporal_encoder(x=y, mask=mask)
        y = rearrange(y, "b T (p D) -> b T p D", p=1)
        return MaskedValue(data=y, mask=mask)
