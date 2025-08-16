import torch
from jedi.inference.gradual.typing import Callable
from torch import nn


# Multimodal Bottleneck Transformer (MBT)
# https://github.com/lucidrains/bottleneck-transformer-pytorch
# I make my own implementation
class BottleneckTransformer(nn.Module):
    def __init__(self, ):
        super(BottleneckTransformer).__init__()

    def forward(self, x):
        pass


class EmbeddingsBridge(nn.Module):
    def __init__(self, source_size: int, target_size: int, kd_size: int, type_embedding: str, use_cls: bool = False):
        """

        :param source_size:
        :param target_size:
        :param kd_size:
        :param type_embedding:
        :param use_cls:
        """
        super(EmbeddingsBridge).__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(source_size),
            nn.Linear(source_size, target_size),
        )

        self.use_cls = use_cls
        self.type_embedding = type_embedding
        self.kd_projection_head = nn.Linear(target_size, kd_size)
        self.logit_scale_kd = nn.Parameter(torch.tensor(2.6592))

    def forward(self, tokens, positions_idx, type_embeddings, time_embeddings: Callable[[int], torch.Tensor]):
        x = self.adapter(tokens)
        x = x + type_embeddings[self.type_embedding] + time_embeddings(positions_idx)

        # KD Branch
        kd_values = x[:, 0] if self.use_cls else x.mean(dim=1)
        kd_values = torch.nn.functional.normalize(self.kd_projection_head(kd_values), dim=-1)
        kd_times = self.logit_scale_kd.exp()

        return x, kd_values, kd_times
