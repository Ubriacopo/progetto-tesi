from typing import Optional

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import nn, einsum

from models.FEEG.layers.base_embedding import FoundationEmbedder
from models.FEEG.layers.kd import KDHead


class PerceiverAttention(nn.Module):
    # Taken from Open-Flamingo
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale: float = dim_head ** -0.5
        self.heads: int = heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * heads * 2, bias=False)
        self.to_out = nn.Linear(dim_head * heads, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latents (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=self.heads)
        q *= self.scale

        # Attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=self.heads)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """
    The PerceiverResampler comes from the flamingo definition and enrichest the Perceiver architecture.
    It has fewer computational complexity and allows us to draw information from timed sequences and condense the info
    where possible. It is good to merge multimodal data later down the stream.
    """

    def __init__(self, dim: int, depth: int, dim_head: int = 64, heads: int = 8, num_latens: int = 64,
                 max_num_media: int = None, max_num_frames: int = None, ff_mult: int = 4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latens, dim))
        # todo vedi di capire meglio
        self.frame_embeddings: Optional[nn.Parameter] = None
        if max_num_frames is not None:
            self.frame_embeddings = nn.Parameter(torch.randn(max_num_frames, dim))

        self.media_time_embeddings: Optional[nn.Parameter] = None
        if max_num_media is not None:
            self.media_time_embeddings = nn.Parameter(torch.randn(max_num_media, 1, dim))

        # Build perceiver layers.
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                self._build_projection_head(dim, ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def _build_projection_head(self, dim: int, ff_mult: int) -> nn.Sequential:
        assert isinstance(ff_mult, int) and ff_mult > 0, "Multiplicator has to be a positive integer"
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x):
        b, T, F, v = x.shape[:4]

        # Add to the resampled x the frame embeddings
        # todo forse non ho capito
        if self.frame_embeddings is not None:
            frame_embs = repeat(self.frame_embeddings[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x += frame_embs

        # Flatten the frame and spatial dimensions
        x = rearrange(x, "b T F v d -> b T (F v) d")
        # Add to the resampled x the time embeddings
        # todo forse non ho capito
        if self.media_time_embeddings is not None:
            x += self.media_time_embeddings[:T]

        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)

        # tuple(Attention, Feed Forward)
        for att, feed_forward in self.layers:
            latents = att(x, latents) + latents
            latents = feed_forward(latents) + latents

        return self.norm(latents)


class PerceiverAdapter(nn.Module):
    def __init__(self, embedder: FoundationEmbedder, resampler_depth: int,
                 target_shape: int, kd: bool = False, kd_size: int = None) -> None:
        """

        :param target_shape:
        :param embedder:
        :param resampler_depth: How many layers of perceiver attention + sequential are desired for the PerceiverResampler
        :param kd: If the current modality uses knowledge distillation
        :param kd_size: To what size to remap the output of the Resampler to match the teacher
        :return:
        """
        super().__init__()
        self.resampler = PerceiverResampler(embedder.output_size, resampler_depth)
        self.kd_head: Optional[KDHead] = None

        if kd:
            # Build KD map
            assert kd_size is not None, "If using KD you should provide kd_size to map to teacher"
            self.kd_head = KDHead(input_dimension=embedder.output_size, output_dimension=kd_size)

        self.reshaper: Optional[nn.Linear] = None
        if target_shape != embedder.output_size:
            self.reshaper = nn.Linear(embedder.output_size, target_shape)

        self.embedder = embedder

    def forward(self, x, use_kd: bool = True) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None:
        if x is None:
            return None  # We can't work when modality is disabled

        e = self.embedder(**x)
        e = self.resampler(e)

        kd_e: Optional[torch.Tensor] = None
        if self.kd_head is not None:
            kd_e = self.kd_head(e)

        if self.reshaper is not None:
            e = self.reshaper(e)

        return e if not use_kd else (e, kd_e)
