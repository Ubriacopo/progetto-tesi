from typing import Optional

import torch
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import nn, einsum

from main.model.layer.perceiver_simple import feed_forward_layer


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner = dim_head * heads
        self.kv_gate = nn.Parameter(torch.tensor(-4.0))  # sigmoid(-4) ≈ 0.0
        self.norm_latents = nn.LayerNorm(dim)
        self.norm_media = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor, key_padding_mask=None):
        """

        :param x:
        :param latents:
        :param key_padding_mask:
        :return:
        """
        b, f, d = x.shape
        n = latents.shape[1]
        assert latents.shape[0] == b and latents.shape[2] == d

        x = self.norm_media(x)
        latents = self.norm_latents(latents)
        kv = torch.cat((x, latents * self.kv_gate), dim=1)

        q = self.to_q(latents)
        k = self.to_k(kv)
        v = self.to_v(kv)

        q = rearrange(q, 'b f (h d) -> b h f d', h=self.heads)
        k, v = rearrange_many((k, v), 'b n (h d) -> b h n d', h=self.heads)

        sim = einsum('b h q d, b h f d -> b h q f', q * self.scale, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        if key_padding_mask is not None:
            # pad mask only on the features part
            assert key_padding_mask.shape == (b, f)
            valid_feat = ~key_padding_mask
            valid_full = torch.cat((valid_feat, torch.ones(b, n, dtype=torch.bool, device=valid_feat.device)), dim=1)
            sim = sim.masked_fill(~valid_full[:, None, None, :], float('-inf'))

        attn = sim.softmax(dim=-1)

        out = einsum('b h q f, b h f d -> b h q d', attn, v)
        out = rearrange(out, 'b h q d -> b q (h d)')
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(self, dim: int, depth: int, dim_head: int = 64, heads: int = 8, num_latents: int = 64,
                 max_num_time_steps: int = None, max_num_frames: int = None, ff_mult: int = 4):
        # We learn the latents
        super().__init__()
        self.dim = dim
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.time_pos_embedding: Optional[nn.Parameter] = None
        if max_num_time_steps is not None:
            self.time_pos_embedding = nn.Parameter(torch.randn(max_num_time_steps, 1, dim))

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                feed_forward_layer(dim=dim, mult=ff_mult),
            ]) for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """

        :param x: [b, t (time frames), N (number frames), F (features), D (dim)]
        :param mask: [b, t]
        :return:
        """
        assert x.dim() == 5 or x.dim() == 4
        if x.dim() == 5:
            x = rearrange(x, 'b t N F D -> b t (N F) D')

        b, t, N, D = x.shape
        assert D == self.dim

        if self.time_pos_embedding is not None:
            tpe = self.time_pos_embedding[:t]
            if mask is not None:
                time_mask = mask[:, :, None, None].to(x.dtype)
                tpe = repeat(tpe, "t n f -> b t n f", b=b) * time_mask
            else:
                tpe = repeat(tpe, "t n f -> b t n f", b=b)

            x += tpe

        # Build KV padding mask over flattened tokens
        x_flat = rearrange(x, "b t N D -> (b t) N D")  # [b*t N D]

        kv_mask = None
        if mask is not None:
            kv_mask = (~mask).reshape(b * t, 1)
            kv_mask = kv_mask.expand(b * t, N)

        latents = repeat(self.latents, 'q d -> (b t) q d', b=b, t=t)  # [B*T, Q, D]
        for attn, ff in self.blocks:
            latents = latents + attn(x_flat, latents, key_padding_mask=kv_mask)
            latents = latents + ff(latents)

        out = self.out_norm(latents)  # [B*T, Q, D]
        return rearrange(out, '(b t) q d -> b t q d', b=b, t=t)
