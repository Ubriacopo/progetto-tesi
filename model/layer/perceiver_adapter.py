from typing import Optional

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import nn, einsum

from model.layer.base import SimpleFeedForward


class PerceiverAttention(nn.Module):

    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        """

        Perceiver Attention take from OpenFlamingo and adapted.

        :param dim: Input latent dimension
        :param dim_head: Attention head latent size
        :param heads: Number of attention heads
        """
        super().__init__()
        self.scale: float = dim_head ** -0.5
        self.heads: int = heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)
        # Linear projections to fit the attention shapes
        self.q = nn.Linear(dim, dim_head * heads, bias=False)
        self.kv = nn.Linear(dim, dim_head * heads * 2, bias=False)
        self.out = nn.Linear(dim_head * heads, dim, bias=False)

    def forward(self, x, latents, mask=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latents (torch.Tensor): latent features
                shape (b, T, n2, D)
            :param mask:
        """

        b, T, Nx, D = x.shape
        _, _, Nz, _ = latents.shape
        assert T == latents.shape[1] and D == latents.shape[-1], \
            f"Mismatch: x(T={T},D={D}) vs latents(T={latents.shape[1]},D={latents.shape[-1]})"

        x, latents = self.norm_media(x), self.norm_latents(latents)

        q = self.q(latents)
        kv = torch.cat((x, latents), dim=-2)
        k, v = self.kv(kv).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=self.heads)
        q *= self.scale

        if mask is None:
            mask = torch.ones((b, T), dtype=torch.bool, device=x.device)
        mask = mask.to(dtype=torch.bool)
        q *= mask[:, None, :, None, None]

        keep_x = mask[:, :, None].expand(b, T, Nx)
        keep_z = mask[:, :, None].expand(b, T, Nz)
        key_keep = torch.cat([keep_x, keep_z], dim=-1)  # (b,T,n1+n2)
        key_keep = key_keep[:, None, :, None, :]  # (b,1,T,1,nk)

        # Attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim.masked_fill(~key_keep, float("-inf"))

        row_has_key = key_keep.any(dim=-1, keepdim=True)
        sim = torch.where(row_has_key, sim, torch.zeros_like(sim))

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        attn = attn * row_has_key

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=self.heads)

        out *= mask[:, :, None, None]
        return self.out(out)


class PerceiverResampler(nn.Module):
    def __init__(self, dim: int, depth: int, dim_head: int = 64, heads: int = 8, num_latents: int = 64,
                 max_num_media: int = None, max_num_frames: int = None, ff_mult: int = 4):
        """
        The PerceiverResampler comes from the Flamingo definition and enrichest the Perceiver architecture.
        It has a lower computational complexity and allows us to draw information from timed sequences.
        It condenses the info where possible.
        It is good to merge multimodal data later down the stream.

        :param dim: Size of the last dimension of the input (latent space).
        :param depth: Number of iterations of Perceiver Attention (nested PerceiverAttention + Projection Head).
        :param dim_head: Size of the heads of Perceiver Attention.
        :param heads: Number of attention heads.
        :param num_latents: Number of latent dimensions for the generation of latens that are passed to the attention.
        :param max_num_media: To create Embeddings for the frames by tagging sequence positioning of media.
        :param max_num_frames: To create Embeddings for the frames by tagging sequence positioning of frames.
        :param ff_mult: Multiplier for the feed forward network to remap the input dim.
        """
        super().__init__()
        # We learn the latents
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

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
                SimpleFeedForward(dim=dim, mult=ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask=None):
        """

        :param mask:
        :param x: Features of the modality in input. It is supposed to be a 5D tensor [b, T, F, v, D] with:
                    - b: is batch size of the input. (Fixed during training).
                    - T: is number of time steps.
                    - F: are the frames fed that compose a single time-step (For ViViT it's 16 for example as it has a temporal stride of 2)
                    - v: Are the generated patches by the embedder previously.
                    - D: Is the final embedding space (e.g. 768 for ViViT)
        :return:
        """
        b, T, F, v = x.shape[:4]

        if self.frame_embeddings is not None:
            # Add the frame embeddings to the input to not lose the temporal alignment information
            x += repeat(self.frame_embeddings[:F], "F d -> b T F v d", b=b, T=T, v=v)

        # Flatten the frame and spatial dimensions
        x = rearrange(x, "b T F v d -> b T (F v) d")

        if self.media_time_embeddings is not None:
            # Add to the resampled x the time embeddings. Mostly won't be used as T tends to be 1.
            x += self.media_time_embeddings[:T]

        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        # Transformer Block
        for att, feed_forward in self.layers:
            latents = att(x, latents, mask=mask) + latents  # Residual network style.
            latents = feed_forward(latents) + latents  # Residual network style.

        return self.norm(latents)
