from typing import Callable, Optional

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


class SimpleFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int) -> None:
        super().__init__()
        assert mult > 0, "Multiplicator has to be a positive integer"
        x, y = dim, dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(x),  # Normalize
            nn.Linear(x, y),  # Map to new shape
            nn.GELU(),  # Non-linearity
            nn.Linear(y, x),  # Rebuild the original shape
        )

    def forward(self, x):
        return self.net(x)


class QueryEEGFormer(nn.Module):
    def __init__(self, in_dim: int, target_dim: int, max_T: int, max_c: int, use_time: bool = False) -> None:
        super().__init__()

        self.projection: Optional[nn.Linear] = None
        if in_dim != target_dim:
            self.projection = nn.Linear(in_dim, target_dim)

        self.time_embeddings: Optional[nn.Embedding] = None
        self.alpha_t: Optional[nn.Parameter] = None
        if self.use_time:  # CBraMod already encodes time in the embeddings.
            self.time_embeddings = nn.Embedding(max_T, target_dim)
            self.alpha_t = nn.Parameter(torch.zeros(1))

        self.channel_embeddings = nn.Embedding(max_c, target_dim)
        # Gated so to not overpower CBraMod’s own PE
        self.alpha_c = nn.Parameter(torch.zeros(1))

        self.net = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, ch, T, D = x.shape
        if self.projection is not None:
            x = self.projection(x)

        time_ids = torch.arange(T, device=x.device)
        channel_ids = torch.arange(ch, device=x.device)

        c = rearrange(self.channel_embeddings(channel_ids), "c D -> () c () D") * self.alpha_c
        t = rearrange(self.time_embeddings(time_ids), "c D -> () () c D") * self.alpha_t

        x = x + c + t
        x = x.permute(0, 2, 1, 3).reshape(b, T * ch, -1)
        return x
