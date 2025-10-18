from __future__ import annotations

import dataclasses

import torch
from einops import rearrange
from einops_exts import rearrange_many
from torch import nn, einsum

from main.model.layer.base import SimpleFeedForward


@dataclasses.dataclass
class GatedXAttentionCustomArgs:
    dim_head: int = 64
    heads: int = 8
    ff_mult: int = 4


class GatedXAttentionBlock(nn.Module):
    def __init__(self, dim: int, dim_latent: int, dim_head: int = 64, heads: int = 6, ff_mult: int = 4):
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

    def forward(self, q, kv, attn_mask=None, q_mask=None, kv_mask=None):
        q = self.attn(q, kv, attn_mask, q_mask, kv_mask) * self.attn_gate.tanh() + q
        q = self.ff(q) * self.ff_gate.tanh() + q
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

        self.norm_qo = nn.LayerNorm(dim)
        self.norm_kvo = nn.LayerNorm(dim)

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
        # Build the query object.
        qo = self.norm_qo(qo)
        kvo = self.norm_kvo(kvo)

        q = self.q(qo)
        k, v = self.kv(kvo).chunk(2, dim=-1)

        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.heads)
        # Rescale the query object
        q *= self.scale

        # Check similarity between key and query
        sim = einsum("... i d, ... j d -> ... i j", q, k)

        NEG_INF = torch.finfo(qo.dtype).min
        # Key padding mask (per token): shape -> (B,1,1,Tkv*n)
        if kv_mask is not None:
            sim = sim.masked_fill(~kv_mask[:, None, None, :], NEG_INF)
        if attn_mask is not None:
            sim = sim.masked_fill(~attn_mask[:, None, :, :], NEG_INF)

        # optional: add soft time bias too (independent of hard mask)
        # if (t_q is not None) and (t_kv is not None):
        #    tb = _time_bias(t_q, t_kv, alpha=self.alpha, dmax=self.dmax)  # [B,Tq,Tk]
        #    # (optional) zero tb on text spans: tb[:, :, s:e] = 0
        #    sim = sim + tb.unsqueeze(1)

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
