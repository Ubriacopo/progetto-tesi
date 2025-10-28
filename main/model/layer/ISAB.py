from typing import Optional

import torch
from einops import rearrange
from torch import nn


class CrossTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dropout: float = 0.0,
                 ffn_hidden_mult: int = 4, batch_first: bool = True):
        super(CrossTransformerBlock, self).__init__()

        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)

        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=batch_first)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden_mult * dim),
            nn.GELU(),  # GELU tends to perform better on Transformers
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden_mult * dim, dim)
        )

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, kv: torch.Tensor, *,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q, kv = self.q_norm(x), self.kv_norm(kv)
        # The reason we had to re-implement the block is q != kv
        # [NORM -> MHA] -> ADD -> [NORM -> FF] -> ADD
        attn, _ = self.mha(q, kv, kv, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        y = attn + x
        # FeedForward -> ADD & NORM
        y = y + self.ffn(self.out_norm(y))
        return y


class PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_seeds: int, ffn_dropout: float = 0.0):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.transformer_block = CrossTransformerBlock(dim, num_heads, ffn_dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B = x.shape[0]
        seeds = self.S.repeat(B, 1, 1, 1)
        key_padding_mask = ~mask if mask is not None else None
        return self.transformer_block(seeds, x, key_padding_mask=key_padding_mask)


class ISAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_I: int, ffn_dropout: float = 0.0):
        """

        :param dim:
        :param num_heads:
        :param num_I:
        :param ffn_dropout:
        """
        super(ISAB, self).__init__()
        # Self-induced query vector backlog
        self.I = nn.Parameter(torch.Tensor(1, num_I, dim))
        nn.init.xavier_uniform_(self.I)

        self.inference_transformer = CrossTransformerBlock(dim, num_heads, ffn_dropout)
        self.latent_inference_transformer = CrossTransformerBlock(dim, num_heads, ffn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        seeds = self.I.repeat(B, 1, 1)

        z = self.inference_transformer(seeds, x)
        z = self.latent_inference_transformer(x, z)
        return z


class ISABPMATimeAdapter(nn.Module):
    def __init__(self, output_size: int,
                 isab_enabled: bool = True,
                 isab_num_heads: int = 8, isab_num_i: int = 10,
                 pma_num_heads: int = 8, pma_num_i: int = 10, ):
        super().__init__()

        self.isab: Optional[ISAB] = None
        if isab_enabled:
            self.isab = ISAB(output_size, isab_num_heads, isab_num_i)

        self.pma = PMA(output_size, pma_num_heads, pma_num_i)

    def forward(self, z: torch.Tensor):
        b = z.shape[0]  # Batch size is leftmost

        z = rearrange(z, 'b T F D -> (b T) F D')
        if self.isab is not None:
            z = self.isab(z)
        z = self.pma(z)

        z = rearrange(z, '(b T) F D -> b T F D', b=b)
        return z
