from typing import Optional

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import nn, einsum, Tensor

from models.FEEG.layers.base_layers import SimpleFeedForward



class MaskedCrossAttention(nn.Module):
    def __init__(self, dim: int, dim_latent: int, dim_head: int = 64, heads: int = 8, only_attend_immediate_media=True):
        """
        Masked cross-attention layers.

        :param dim: Final shape of the query vector space
        :param dim_latent: Final shape of the kv vector space
        :param dim_head: Features for each attention head
        :param heads: Number of attention heads
        :param only_attend_immediate_media:
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads: int = heads
        self.norm = nn.LayerNorm(dim)

        self.q = nn.Linear(dim, dim_head * heads, bias=False)
        self.kv = nn.Linear(dim_latent, dim_head * heads * 2, bias=False)
        self.out = nn.Linear(dim_head * heads, dim, bias=False)

        # TODO Capire
        # Whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, qo, kvo, media_locations=None, use_cached_media=False):
        """
        Args:
            qo (torch.Tensor): Main modality wanted features
                shape (B, T1, D1)
            kvo (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            use_cached_media: bool
                If true, treat all of x as if they occur after the last media
                registered in media_locations. T_txt does not need to exactly
                equal media_locations.shape[1] in this case
        """

        if not use_cached_media:
            assert (media_locations.shape[1] == qo.shape[1]), \
                f"media_location.shape is {media_locations.shape} but x.shape is {qo.shape}"

        Tq = qo.shape[1]  # Time steps of query
        _, Tkv, n = kvo.shape[:3]  # Time steps of kv
        # Build the query object.
        q = self.q(self.norm(qo))

        kvo = rearrange(kvo, "b t n d -> b (t n) d")
        k, v = self.kv(kvo).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.heads)
        # Rescale the query object
        q *= self.scale

        # Check similarity between key and query
        sim = einsum("... i d, ... j d -> ... i j", q, k)

        q_time: Optional[Tensor] = None
        if media_locations is not None:
            kv_time = torch.arange(Tkv, device=qo.device) + 1
            if use_cached_media:
                q_time = repeat(torch.count_nonzero(media_locations, dim=1), "b -> b i", i=Tq)
            else:
                q_time = media_locations.cumsum(dim=-1)

            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge
            q_to_kv_mask = mask_op(rearrange(q_time, "b i -> b 1 i 1"), repeat(kv_time, "j -> 1 1 1 (j n)", n=n))

            sim = sim.masked_fill(~q_to_kv_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if media_locations is not None and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            q_without_kv_mask = q_time == 0
            q_without_kv_mask = rearrange(q_without_kv_mask, "b i -> b 1 i 1")
            attn = attn.masked_fill(q_without_kv_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, dim_latent: int, dim_head: int = 64, heads: int = 6,
                 ff_mult: int = 4, only_attend_immediate_media: bool = True):
        """

        :param dim:
        :param dim_latent:
        :param dim_head: Number of features for each attention head
        :param heads: Number of heads of masked cross attention
        :param ff_mult: Multiplier for the feed forward structure
        :param only_attend_immediate_media:
        """
        super().__init__()
        # First call
        self.attn = MaskedCrossAttention(dim, dim_latent, dim_head, heads, only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([.1]))

        self.ff = SimpleFeedForward(dim=dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([.1]))

    def forward(self, q, kv, media_locations=None, use_cached_media=False, ):
        use_cached_media = True  # To see if it works at 0. Poi sara da fare
        q = self.attn(q, kv, media_locations, use_cached_media, ) * self.attn_gate.tanh() + q
        q = self.ff(q) * self.ff_gate.tanh() + q

        return q
