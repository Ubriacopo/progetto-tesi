from typing import Callable, Optional

from einops import rearrange, repeat
from torch import einsum, Tensor
import torch
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


# gated cross attention adapted for EEG queries
class MaskedCrossAttention(nn.Module):
    # todo adapt
    def __init__(self, dim: int, dim_latent: int, dim_head: int = 64, heads: int = 8, only_attend_immediate_media=True):
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
            assert (
                    media_locations.shape[1] == qo.shape[1]
            ), f"media_location.shape is {media_locations.shape} but x.shape is {qo.shape}"

        # q is the wrong shape it seems has to be 3d
        TQ = qo.shape[1]
        _, TKV, n = kvo.shape[:3]

        # TODO For now media_locations is always None. Fix it and see to fit with flamingo
        # Build the query object.
        qo = self.norm(qo)
        q = self.to_q(qo)

        kvo = rearrange(kvo, "b t n d -> b (t n) d")
        k, v = self.to_kv(kvo).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.heads)
        q *= self.scale  # Rescale the query
        # Check similarity between key and query
        sim = einsum("... i d, ... j d -> ... i j", q, k)

        q_time: Optional[Tensor] = None
        if media_locations is not None:
            kv_time = torch.arange(TKV, device=qo.device) + 1
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
    def __init__(self, dim: int, dim_latent, dim_head=64, heads=8, ff_mult=4,
                 only_attend_immediate_media=True, time_embedding_max_size: Optional[int] = 5,
                 channel_embedding_max_size: Optional[int] = 17):
        super().__init__()
        self.attn = MaskedCrossAttention(dim, dim_latent, dim_head, heads, only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([.1]))

        # If the sequence we are modelling has time we loose it rep. We simply add the
        # embedding inside the q media. Same will occour for the channels.
        self.time_embeddings: Optional[nn.Embedding] = None
        if time_embedding_max_size is not None:
            self.time_embeddings = nn.Embedding(time_embedding_max_size, dim)

        self.channel_embeddings: Optional[nn.Embedding] = None
        if channel_embedding_max_size is not None:
            self.channel_embeddings = nn.Embedding(channel_embedding_max_size, dim)

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


class AuxiliaryEEGEncoding(nn.Module):
    def __init__(self, dim: int, max_time_embedding_size: int, max_channel_embedding_size: int):
        super().__init__()
        self.time_embeddings = nn.Embedding(max_time_embedding_size, dim)
        self.channel_embeddings = nn.Embedding(max_channel_embedding_size, dim)

    def forward(self, x):
        b, c, T, D = x.shape
        x = rearrange(x, "b c T D -> b (c T) D")
        time_ids = torch.arange(T, device=x.device).repeat_interleave(c)
        channel_ids = torch.arange(c, device=x.device).repeat(c)
        return x + self.time_embeddings[time_ids], self.channel_embeddings[channel_ids]
