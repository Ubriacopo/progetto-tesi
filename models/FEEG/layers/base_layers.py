from typing import Callable

from einops import rearrange
import torch
from torch import nn


class AuxiliaryEEGEncoder(nn.Module):
    def __init__(self, dim: int, max_time_embedding_size: int, max_channel_embedding_size: int):
        """
        Module to encode EEG channels among the time sequence inside the reduced dimensionality embeddings.
        :param dim: The latent space dimension
        :param max_time_embedding_size: Maximum time sequence (thus its largest index in embeddings).
        :param max_channel_embedding_size: Maximum channel of the input (thus its largest index in embeddings). EEG data varies, being 17-21 the usual values.
        """
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
    def __init__(self, dim: int, modality_mappings: dict[str, int], weights=None):
        """
        Adds to the input embeddings a weight vector indicating the modality of the record.
        :param dim: Latent space dimension
        :param modality_mappings: Map string -> index . It maps the modality with the embedding row in the matrix.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        max_embedding_rows = max(modality_mappings.values())
        self.modal_embeddings = nn.Embedding(max_embedding_rows, dim)
        # Suppose the weights are already trained. We keep it and load it. This is the reason to get a dictionary
        # instead of a str set as the order and indexes may vary with time.
        if weights is not None:
            self.modal_embeddings.load_state_dict(weights)

        self.modality_mappings = modality_mappings

    def forward(self, x, modality: str):
        if x is None: return None
        idx = torch.tensor(self.modality_mappings[modality], dtype=torch.long, device=x.device)
        return self.norm(x) + self.modal_embeddings(idx).view(1, 1, -1)
