from typing import Callable, Optional

from einops import rearrange, repeat
from torch import einsum, Tensor
import torch
from transformer import *
from einops_exts import rearrange_many
from torch import nn


class KDHead(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int):
        super(KDHead, self).__init__()
        # Output shape is teacher shape
        self.projection = nn.Linear(input_dimension, output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.projection(x), p=2, dim=-1)


class EmbeddingsBridge(nn.Module):
    def __init__(self, source_size: int, target_size: int, kd_size: int, type_embedding: str):
        """
        Bridges the embedding layer
        :param source_size:
        :param target_size:
        :param kd_size:
        :param type_embedding:
        """
        super(EmbeddingsBridge, self).__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(source_size),
            nn.Linear(source_size, target_size),
        )

        self.type_embedding = type_embedding
        self.kd_projection_head = nn.Linear(target_size, kd_size)
        self.logit_scale_kd = nn.Parameter(torch.tensor(2.6592))

    def forward(self, tokens, positions_idx, type_embeddings, time_embeddings: Callable[[int], torch.Tensor]):
        x = self.adapter(tokens)
        x = x + type_embeddings[self.type_embedding] + time_embeddings(positions_idx)

        # KD Branch
        kd_values = torch.nn.functional.normalize(self.kd_projection_head(x.mean(dim=1)), dim=-1)
        kd_times = self.logit_scale_kd.exp()

        return x, kd_values, kd_times


# todo studia
class PerceiverAttention(nn.Module):
    # Taken from Open-Flamingo
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale: float = dim_head ** -0.5
        self.heads: int = heads

        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

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
        q = q * self.scale

        # Attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=self.heads)
        return self.to_out(out)


# TODO STUDIA PER CAPIRE COSA FA
class PerceiverResampler(nn.Module):
    """
    The PerceiverResampler comes from the flamingo definition and enrichest the Perceiver architecture.
    It has fewer computational complexity and allows us to draw information from timed sequences and condense the info
    where possible. It is good to merge multimodal data later down the stream.
    """

    def __init__(self, dim: int, depth: int, dim_head: int = 64, heads: int = 8, num_latens: int = 64,
                 max_num_media=None, max_num_frames=None, ff_mult: int = 4):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(num_latens, dim))

        self.frame_embeddings: Optional[nn.Parameter] = None
        if max_num_frames is not None:
            self.frame_embeddings = nn.Parameter(torch.randn(max_num_frames, dim))

        self.media_time_embeddings: Optional[nn.Parameter] = None
        if max_num_media is not None:
            self.media_time_embeddings = nn.Parameter(torch.randn(max_num_media, 1, dim))

        # Build perceiver layers.
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            inner_dim = int(dim * ff_mult)
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, inner_dim),
                    nn.GELU(),
                    nn.Linear(inner_dim, dim),
                )
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        b, T, F, v = x.shape[:4]
        if self.frame_embeddings is not None:
            frame_embs = repeat(self.frame_embeddings[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x += frame_embs
        x = rearrange(x, "b T F v d -> b T (F v) d")  # Flatten the frame and spatial dimensions
        if self.media_time_embeddings is not None:
            x += self.media_time_embeddings[:T]

        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)

        # tuple(Attention, Feed Forward)
        for att, feed_forward in self.layers:
            latents = att(x, latents) + latents
            latents = feed_forward(latents) + latents

        return self.norm(latents)


# gated cross attention
class MaskedCrossAttention(nn.Module):
    # todo adapt
    def __init__(self, dim: int, dim_latent, dim_head=64, heads=8, only_attend_immediate_media=True):
        super().__init__()
        # TODO Non vista la costruzione
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_latent, inner_dim * 2, bias=False)
        self.output = nn.Linear(inner_dim, dim, bias=False)

        # TODO Capire
        # Whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None, use_cached_media=False):
        """
        Args:
            x (torch.Tensor): Main modality wanted features
                shape (B, T1, D1)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            use_cached_media: bool
                If true, treat all of x as if they occur after the last media
                registered in media_locations. T_txt does not need to exactly
                equal media_locations.shape[1] in this case
        """

        if not use_cached_media:
            assert (
                    media_locations.shape[1] == x.shape[1]
            ), f"media_location.shape is {media_locations.shape} but x.shape is {x.shape}"
        q_object = x
        kv_object = media
        # q is the wrong shape it seems has to be 3d
        TQ = q_object.shape[1]  # 22 from Tensor([1,22,4,200])
        _, TKV, n = kv_object.shape[:3]  # (1,33,64) from Tensor([1,33,64,768])

        q_object = self.norm(q_object)
        q = self.to_q(q_object)
        # todo gioca con le forme solo qui non prima
        kv_object = rearrange(kv_object, "b t n d -> b (t n) d")
        k, v = self.to_kv(kv_object).chunk(2, dim=-1)
        # TODO: Checkpoint
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.heads)
        q *= self.scale  # Rescale the query

        # Check similarity between key and query
        sim = einsum("... i d, ... j d -> ... i j", q, k)

        q_time: Optional[Tensor] = None
        kv_time: Optional[Tensor] = None  # todo vedi se giusto
        if kv_object is not None:
            kv_time = torch.arange(TKV, device=x.device) + 1
            if use_cached_media:
                q_time = repeat(torch.count_nonzero(media_locations, dim=1), "b -> b i", i=TQ)
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
        return self.output(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, dim_latent, dim_head=64, heads=8, ff_mult=4, only_attend_immediate_media=True, ):
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_latent=dim_latent, dim_head=dim_head, heads=heads,
                                         only_attend_immediate_media=only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([.1]))

        inner_dim = dim * ff_mult
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim),
        )

        self.ff_gate = nn.Parameter(torch.tensor([.1]))

    def forward(self, q, kv, media_locations=None, use_cached_media=False, ):
        use_cached_media = True  # To see if it works at 0. Poi sara da fare
        q = self.attn(q, kv, media_locations, use_cached_media, ) * self.attn_gate.tanh() + q
        q = self.ff(q) * self.ff_gate.tanh() + q  # Residual network
        return q
