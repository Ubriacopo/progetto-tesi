from dataclasses import dataclass
from typing import Optional

import torch.nn.functional as F
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


# todo verifica matematica e metti in utils
# todo: Errore sta in perceiver resampler pare!
@torch.no_grad()
def batch_stats_5d(X: torch.Tensor,
                   mask: torch.Tensor | None = None,
                   reduce_axes=(1, 2, 3), max_dim=3):  # pool over T,F,P by default
    """
    X:    [B, T, F, P, D]
    mask: [B, T, F] or [B, T, F, P] (True = valid)
    """
    B = X.size(0)
    X = X.to(torch.float32)

    if mask is not None:
        # broadcast mask to [B,T,F,P,1]
        if mask.dim() == max_dim:  # [B,T,F]
            mask = mask.unsqueeze(-1)  # [B,T,F,1]
        mask = mask.to(X.device)
        w = mask.unsqueeze(-1).to(X.dtype)  # [B,T,F,P,1]

        # masked numerator/denominator
        num = (X * w).sum(dim=reduce_axes)  # -> [B, D]
        den = w.sum(dim=reduce_axes)  # -> [B, 1]
        den = den.clamp_min(1e-6)
        Xb = num / den  # [B, D]
    else:
        Xb = X.mean(dim=reduce_axes)  # [B, D]

    # similarity stats across the batch
    Xn = F.normalize(Xb, dim=-1)
    S = Xn @ Xn.T  # [B, B]
    diag = S.diag().mean().item()
    off = (S.sum() - S.diag().sum()) / (S.numel() - B)

    # rank-1 dominance check
    Xm = Xn - Xn.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Xm)  # faster than full svd
    rank1_ratio = (s[0] / (s.sum() + 1e-9)).item()

    return dict(
        diag=float(diag),
        off=float(off),
        gap=float(diag - off),
        S_min=float(S.min()),
        S_max=float(S.max()),
        rank1_ratio=rank1_ratio,
        across_batch_std=float(Xb.std(0).mean())
    )


class AudioAdapter(nn.Module):
    def __init__(self, perceiver_config: PerceiverResamplerConfig, project_out_size: int = None, ):
        super().__init__()
        self.resampler = PerceiverResampler(**perceiver_config.__dict__)
        self.projection: Optional[nn.Module] = None
        if project_out_size is not None and project_out_size != perceiver_config.dim:
            self.projection = nn.Linear(perceiver_config.dim, project_out_size)

    def forward(self, x: torch.Tensor, mask=None) -> MaskedValue:
        # Audio embeddings have no further decomposition. We simply add a dim to fit requirements
        x = rearrange(x, "b T (F p) D -> b T F p D", F=1)

        # BEFORE resampler (raw audio features you feed in)
        stats_pre = batch_stats_5d(
            x, mask=repeat(mask, "b T -> b T F p", F=1, p=x.shape[-2])
        )  # e.g., pooled or CLS before Perceiver
        # AFTER resampler + your pooling to [B,D] that goes into loss

        y = self.resampler(x=x, mask=mask)
        if self.projection is not None:
            y = self.projection(y)

        stats_post = batch_stats_5d(
            y, mask=repeat(mask, "b T -> b T p", p=y.shape[-2]), max_dim=2, reduce_axes=(1, 2)
        )  # the exact zb you pass to InfoNCE
        print("PRE :", stats_pre)
        print("POST:", stats_post)

        return MaskedValue(data=y, mask=mask)


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
