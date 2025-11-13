from __future__ import annotations

import dataclasses
from typing import Optional

import torch
from einops import rearrange
from einops_exts import rearrange_many
from torch import nn, einsum

from main.model.neegavi.blocks import SimpleFeedForward


@dataclasses.dataclass
class GatedXAttentionCustomArgs:
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4


class GatedXAttentionBlock(nn.Module):
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
        self.attn_gate = nn.Parameter(torch.tensor([0.]))
        self.ff = SimpleFeedForward(dim=dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)

        self.self_attn: Optional[nn.Module] = None
        self.self_attn_gate: Optional[nn.Parameter] = None

        if with_self_attn:
            self.norm_self_attn = nn.LayerNorm(dim)
            self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=2, batch_first=True)
            self.self_attn_gate = nn.Parameter(torch.tensor([1.]))

    def forward(self, q, kv, attn_mask=None, q_mask=None, kv_mask=None):
        # Pre-LN + Cross modality attention
        norm_q = self.norm_q(q)
        norm_kv = self.norm_kv(kv)
        q = q + self.attn(norm_q, norm_kv, attn_mask, q_mask, kv_mask) * self.attn_gate.tanh()

        if self.self_attn is not None:
            # Similar to how Flamingo works just that this self attn is not frozen but learnt.
            # Also respect the convention of torch of passing mask with True where ignore.
            norm_q = self.norm_self_attn(q)
            out, _ = self.self_attn(norm_q, norm_q, norm_q, key_padding_mask=~q_mask, need_weights=False)
            q = q + self.self_attn_gate.tanh() * out

        norm_q = self.norm_ff(q)
        q = q + self.ff(norm_q) * self.ff_gate.tanh()

        return q

    def old_forward(self, q, kv, attn_mask=None, q_mask=None, kv_mask=None):
        # Pre-LN
        norm_q = self.norm_q(q)
        norm_kv = self.norm_kv(kv)

        # Cross modality attention
        q = q + self.attn(norm_q, norm_kv, attn_mask, q_mask, kv_mask) * self.attn_gate.tanh()
        norm_q = self.norm_ff(q)
        q = q + self.ff(norm_q) * self.ff_gate.tanh()

        if self.self_attn is not None:
            # Similar to how Flamingo works just that this self attn is not frozen but learnt.
            # Also respect the convention of torch of passing mask with True where ignore.
            norm_q = self.norm_self_attn(q)
            out, _ = self.self_attn(norm_q, norm_q, norm_q, key_padding_mask=~q_mask, need_weights=False)
            q = q + self.self_attn_gate.tanh() * out

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
            attn_mask: boolean mask identifying the media tokens in x
            kv_mask:
                shape (B, T)
            q_mask:
                shape (B, T)
        """
        _, Tkv, n = kvo.shape[:3]  # Time steps of kv
        q = self.q(qo)
        k, v = self.kv(kvo).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.heads)
        # Rescale the query object
        q *= self.scale

        # Check similarity between key and query
        sim = einsum("... i d, ... j d -> ... i j", q, k)
        # Key padding mask (per token): shape -> (B,1,1,Tkv*n)
        if kv_mask is not None:
            mask = ~kv_mask[:, None, None, :]  # shape [B,1,1,S], bool
            sim.masked_fill_(mask, torch.finfo(sim.dtype).min)
        if attn_mask is not None:
            # sim = sim.masked_fill(~attn_mask[:, None, :, :], neg_inf)
            mask = ~attn_mask[:, None, :, :]  # shape [B,1,1,S], bool
            sim.masked_fill_(mask, torch.finfo(sim.dtype).min)

        # Guard rows that are fully -inf (all keys masked)
        row_has_key = torch.isfinite(sim).any(dim=-1, keepdim=True)  # (B,H,Tq,1)
        sim = torch.where(row_has_key, sim, torch.zeros_like(sim))

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        attn = attn * row_has_key

        # Zero invalid query steps defensively
        if q_mask is not None:
            attn = attn.masked_fill(~q_mask[:, None, :, None], 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.out(out)
