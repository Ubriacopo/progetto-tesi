# Not fit for late fusion like we want it.

# From official repository by Google
# https://github.com/google-research/scenic/blob/main/scenic/projects/mbt/model.py
# Implementation drawn from JAX rewritten in Torch
from typing import Tuple

from torch import nn, Tensor


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, mpl_dimension: int,
                 num_heads: int, dropout: float | None = None,
                 attention_dropout: float | None = None):
        """

        :param dim:
        :param mpl_dimension: Dimension of the mlp on top of attention block.
        :param num_heads: Number of heads.
        :param dropout:
        :param attention_dropout:
        """
        super(EncoderBlock, self).__init__()

        self.attention = nn.Sequential(
            nn.LayerNorm(dim),
            nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attention_dropout, batch_first=True),
            nn.Dropout(dropout)
        )

        # MLP block todo capire
        self.mlpBlock = nn.Sequential(
            # Suppose to create the bottlenecks
            nn.LayerNorm(mpl_dimension),
            nn.Linear(mpl_dimension, dim),
            nn.Linear(dim, mpl_dimension),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Attention block computation
        weights = self.attention(x)
        # drop_pattern = self.get_drop_pattern(weights, True)
        weights = weights + x  # * (1. - drop_pattern) + x

        # MLP block
        y = self.mlpBlock(weights)
        y = y + weights  # * (1 - drop_pattern) + weights
        return y


class Encoder(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, number_layers: int, num_heads: int, dropout: float | None = None,
                 modality_fusion: Tuple[str] = ('todo',)):
        super(Encoder, self).__init__()
        # Args initialization
        self.dim: int = dim
        self.mlp_dim: int = mlp_dim
        self.num_heads: int = num_heads
        self.dropout = dropout
        self.modality_fusion = modality_fusion

        self.encoders = {}
        self.layers = [self.build_layer(idx) for idx in range(self.num_layers)]
        self.layers = nn.ModuleList()

    def build_layer(self, layer_idx: int) -> nn.Module:
        encoders = {}
        # todo allora se ho capito ogni modality (video audio text deve avere il suo encoder che
        #       sarebbe un transformer. Ma io questo lo faccio gia  altrove quindi non mi serve?
        encoder = EncoderBlock(self.dim, self.mlp_dim, self.num_heads, self.dropout)

    def forward(self, x: dict) -> Tensor:
        for modality in self.modality_fusion:
            # TODO vedi di capire
            # x[modality] = add_positional_embeddings()
            pass


class MBT(nn.Module):
    def __init__(self):
        super(MBT, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass
