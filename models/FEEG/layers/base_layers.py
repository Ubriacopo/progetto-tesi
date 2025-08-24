from typing import Callable

from einops import rearrange
import torch
from torch import nn


# todo vedi se serve
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


class AuxiliaryEEGEncoder(nn.Module):
    def __init__(self, dim: int, max_time_embedding_size: int, max_channel_embedding_size: int):
        super().__init__()
        self.time_embeddings = nn.Embedding(max_time_embedding_size, dim)
        self.channel_embeddings = nn.Embedding(max_channel_embedding_size, dim)

    def forward(self, x):
        b, c, T, D = x.shape
        x = rearrange(x, "b c T D -> b (c T) D")
        time_ids = torch.arange(T, device=x.device).repeat_interleave(c)
        channel_ids = torch.arange(c, device=x.device).repeat(T)
        return x + self.time_embeddings.weight[time_ids] + self.channel_embeddings.weight[channel_ids]


class ModalContextEncoder(nn.Module):
    def __init__(self, dim: int, modality_mappings: dict[str, int]):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.modal_embeddings = nn.Embedding(len(modality_mappings), dim)

        self.modality_mappings = modality_mappings

    def forward(self, x, modality: str):
        if x is None: return None
        idx = torch.tensor(self.modality_mappings[modality], dtype=torch.long, device=x.device)
        return self.norm(x) + self.modal_embeddings(idx).view(1, 1, -1)
