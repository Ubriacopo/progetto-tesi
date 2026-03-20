from __future__ import annotations

import dataclasses
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from einops_exts import rearrange_many
from torch import nn

from main.model.blocks.attention import AbstractAttentionBlock
from main.model.blocks.feed_forward import SimpleFeedForward


@dataclasses.dataclass
class GatedXAttentionCustomArgs:
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4


class GatedXAttentionBlock(AbstractAttentionBlock):
    def __init__(self, dim: int, dim_latent: int, dim_head: int = 64, heads: int = 6, ff_mult: int = 4,
                 with_self_attn: bool = True):
        """

        :param dim:
        :param dim_latent:
        :param dim_head: Number of features for each attention head
        :param heads: Number of heads of masked cross attention
        :param ff_mult: Multiplier for the feed forward structure
        """
        super().__init__()
        # First call
        self.attn = MaskedCrossAttention(dim=dim, dim_latent=dim_latent, dim_head=dim_head, heads=heads)
        self.attn_gate = nn.Parameter(torch.tensor([1.]))
        self.ff = SimpleFeedForward(dim=dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([1.]))

        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)

        self.self_attn: Optional[nn.MultiheadAttention] = None
        self.self_attn_gate: Optional[nn.Parameter] = None

        if with_self_attn:
            self.self_attn_norm = nn.LayerNorm(dim)
            self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=2, batch_first=True)
            self.self_attn_gate = nn.Parameter(torch.tensor([1.]))

    def forward(self, q, kv, attn_mask=None, q_mask=None, kv_mask=None):
        # Pre-LN + Cross modality attention
        norm_q = self.q_norm(q)
        norm_kv = self.kv_norm(kv)
        q = q + self.attn(norm_q, norm_kv, attn_mask, q_mask,
                          kv_mask) * self.attn_gate.sigmoid()  # TODO: Check if this helps. I changed from tanh to sigmoid

        if self.self_attn is not None:
            # Similar to how Flamingo works just that this self attn is not frozen but learnt.
            # Also respect the convention of torch of passing mask with True where ignore.
            norm_q = self.self_attn_norm(q)
            out, _ = self.self_attn(norm_q, norm_q, norm_q, key_padding_mask=~q_mask, need_weights=False)
            q = q + self.self_attn_gate.tanh() * out

        norm_q = self.ff_norm(q)
        q = q + self.ff(norm_q) * self.ff_gate.tanh()
        return q


class MaskedCrossAttention(nn.Module):
    def __init__(self, dim: int, dim_latent: int, dim_head: int = 64, heads: int = 8):
        """
        Masked cross-attention layers.

        :param dim: Final shape of the query vector space
        :param dim_latent: Final shape of the kv vector space
        :param dim_head: Features for each attention head
        :param heads: Number of attention heads
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads: int = heads
        self.q = nn.Linear(dim, dim_head * heads, bias=False)
        self.kv = nn.Linear(dim_latent, dim_head * heads * 2, bias=False)
        self.out = nn.Linear(dim_head * heads, dim, bias=False)

    def forward(self, qo, kvo, attn_mask=None, q_mask=None, kv_mask=None):
        """
        Args:
            qo (torch.Tensor): Main modality wanted features
                shape (B, T, D1)
            kvo (torch.Tensor): Fused features
                shape (B, T, D2)
            attn_mask: boolean mask identifying the media tokens in x, True on attend steps False on ignore ones
            kv_mask: True on attend steps False on ignore ones
                shape (B, T)
            q_mask:
                shape (B, T)
        """
        # If no supports are passed the attention mechanism is ignored.
        if kv_mask is not None and kv_mask.sum() == 0:
            return torch.zeros_like(qo, device=kvo.device)

        q = self.q(qo)
        k, v = self.kv(kvo).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.heads)
        q *= self.scale

        full_mask = None
        # Build combined mask (True = attend)
        if kv_mask is not None:
            # kv_mask: [B, Tkv], True=valid
            full_mask = kv_mask[:, None, None, :]

        if attn_mask is not None:
            # attn_mask expected: [B, Tq, Tkv], True=valid
            am = attn_mask[:, None, :, :]  # [B,1,Tq,Tkv], True=valid
            full_mask = am if full_mask is None else (full_mask & am)

        row_has_key = None
        if full_mask is not None:
            row_has_key = full_mask.any(dim=-1, keepdim=True)  # [B,1,Tq,1]

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=full_mask, dropout_p=0.0, is_causal=False)

        if row_has_key is not None:
            out = out * row_has_key.to(out.dtype)

        if q_mask is not None:
            out = out.masked_fill(~q_mask[:, None, :, None], 0.0)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.out(out)


# todo classic style builder
class GatedXAttentionFactory:
    def __init__(self, dim: int, latent_dim: int):
        """
        This factory creates Gated XAttention layers based on given configurations.

        :param dim: The output size of the gated XAttention layer
        :param latent_dim: The input size of the latent dim (KV)
        """
        self.dim: int = dim
        self.latent_dim: int = latent_dim
        self.default_configuration = GatedXAttentionCustomArgs()

    def build(self, layers: int | list[GatedXAttentionCustomArgs]) \
            -> list[GatedXAttentionBlock]:
        if isinstance(layers, list) and len(layers) > 1 and isinstance(layers[0], GatedXAttentionCustomArgs):
            return [GatedXAttentionBlock(self.dim, self.latent_dim, **asdict(c)) for c in layers]
        if isinstance(layers, int):
            config = asdict(self.default_configuration)
            return [GatedXAttentionBlock(self.dim, self.latent_dim, **config) for _ in range(layers)]

        raise ValueError("Either layers or overrides must be specified")
