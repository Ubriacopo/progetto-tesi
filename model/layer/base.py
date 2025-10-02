import torch
from torch import nn
from torch.nn.functional import softmax


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


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dimension: int) -> None:
        """
        Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
        https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
        It gives each token of the input an attention weight for relevance.

        :param input_dimension: Hidden size
        """
        super().__init__()
        self.W = nn.Linear(input_dimension, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = softmax(self.W(x).squeeze(-1)).unsqueeze(-1)
        return torch.sum(x * attn, dim=1)


class ModalContextEncoder(nn.Module):
    def __init__(self, dim: int, modality_mappings: dict[str, int], weights=None):
        """
        Adds to the input embeddings a weight vector indicating the modality of the record.
        :param dim: Latent space dimension
        :param modality_mappings: Map string -> index . It maps the modality with the embedding row in the matrix.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        max_embedding_rows = max(modality_mappings.values()) + 1  # Indexing start at 0
        self.modal_embeddings = nn.Embedding(max_embedding_rows, dim)
        # Suppose the weights are already trained. We keep it and load it. This is the reason to get a dictionary
        # instead of a str set as the order and indexes may vary with time.
        if weights is not None:
            self.modal_embeddings.load_state_dict(weights)
        self.modality_mappings = modality_mappings

    def forward(self, x: torch.Tensor, modality: str):
        if x is None: return None
        idx = torch.tensor(self.modality_mappings[modality], dtype=torch.long, device=x.device)
        return self.norm(x) + self.modal_embeddings(idx).view(1, 1, 1, -1)


class TemporalEncoder(nn.Module):
    def __init__(self, dim=384, layers=1, heads=8, dropout=0.0):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=layers)
        self.pos = nn.Parameter(torch.randn(1, 512, dim))  # or sinusoidal

    def forward(self, x, mask=None):  # x: (B,T,D), mask: (B,T) bool True=valid
        T = x.size(1)
        x = x + self.pos[:, :T]
        kpm = ~mask if mask is not None else None
        return self.enc(x, src_key_padding_mask=kpm)  # -> (B,T,D)
