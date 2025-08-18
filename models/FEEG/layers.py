from typing import Callable

from einops import rearrange
from torch import einsum

from transformer import *


class KDHead(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int):
        super(KDHead).__init__()
        # Output shape is teacher shape
        self.proj = nn.Linear(input_dimension, output_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.proj(x), p=2, dim=-1)


# todo vedi come funziona
class MultiScale_Bottleneck_Transformer(nn.Module):
    def __init__(self, hid_dim, n_head, dropout, n_bottleneck=8, bottleneck_std=0.15):
        super(MultiScale_Bottleneck_Transformer, self).__init__()
        self.n_layers = int(math.log2(n_bottleneck)) + 1
        self.sma = nn.ModuleList([
            TransformerLayer(hid_dim, MultiHeadAttention(h=n_head, d_model=hid_dim),
                             PositionalFeedForward(hid_dim, hid_dim), dropout=dropout)
            for _ in range(self.n_layers)])
        self.decoder = TransformerLayer(hid_dim, MultiHeadAttention(h=n_head, d_model=hid_dim),
                                        PositionalFeedForward(hid_dim, hid_dim), dropout=dropout)
        self.bottleneck_list = nn.ParameterList([
            nn.Parameter(
                nn.init.normal_(torch.zeros(1, int(n_bottleneck / (2 ** layer_i)), hid_dim).cuda(), std=bottleneck_std))
            for layer_i in range(self.n_layers)])

    def forward(self, m_a, m_b):
        n_batch = m_a.shape[0]
        n_modality = m_a.shape[1]
        bottleneck = self.bottleneck_list[0]
        bottleneck = bottleneck.repeat([n_batch, 1, 1])
        m_a_in, m_b_in = m_a, m_b

        for layer_i in range(self.n_layers):
            m_a_cat = torch.cat([m_a_in, bottleneck], dim=1)
            m_a_cat = self.sma[layer_i](m_a_cat, m_a_cat, m_a_cat)
            m_a_in = m_a_cat[:, :n_modality, :]
            m_a_bottleneck = m_a_cat[:, n_modality:, :]

            if layer_i < self.n_layers - 1:
                next_bottleneck = self.bottleneck_list[layer_i + 1]
                next_bottleneck = next_bottleneck.repeat([n_batch, 1, 1])
                bottleneck = self.decoder(next_bottleneck, m_a_bottleneck, m_a_bottleneck)

            m_b_cat = torch.cat([m_b_in, m_a_bottleneck], dim=1)
            m_b_cat = self.sma[layer_i](m_b_cat, m_b_cat, m_b_cat)
            m_b_in = m_b_cat[:, :n_modality, :]

        return m_b_in, m_a_bottleneck


class RelativePositionalEmbeddings(nn.Module):
    pass


class Attention(nn.Module):
    def __init__(self, dim, feature_map_size: int, dim_head: int, num_heads: int = 8):
        super(Attention).__init__()
        self.num_heads = num_heads
        self.scale = dim_head * self.num_heads

        inner_dim = num_heads * dim
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.pos_emb = RelativePositionalEmbeddings(feature_map_size, dim_head)

    def forward(self, x):
        heads = self.num_heads
        b, c, h, w = x.shape

        q, k, v = self.to_qkvx(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), (q, k, v))

        # Rescale query vector
        q *= self.scale

        sim: torch.Tensor = einsum('b h i d, b h j d -> b h i j', q, k)
        sim += self.pos_emb(q)

        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return out


# Multimodal Bottleneck Transformer (MBT)
# https://github.com/lucidrains/bottleneck-transformer-pytorch
# I make my own implementation
# https://github.com/shengyangsun/MSBT
class BottleneckTransformer(nn.Module):
    pass


class EmbeddingsBridge(nn.Module):
    def __init__(self, source_size: int, target_size: int, kd_size: int, type_embedding: str):
        """
        Bridges the embedding layer
        :param source_size:
        :param target_size:
        :param kd_size:
        :param type_embedding:
        """
        super(EmbeddingsBridge).__init__()
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
