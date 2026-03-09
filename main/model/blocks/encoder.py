from __future__ import annotations

import torch
from torch import nn

from main.model.blocks.time_masked import TimeMaskSwitchable, TimeMaskSwitchableProperties


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
        self.modal_embedding = nn.Embedding(max_embedding_rows, dim)
        # Suppose the weights are already trained. We keep it and load it. This is the reason to get a dictionary
        # instead of a str set as the order and indexes may vary with time.
        if weights is not None:
            self.modal_embedding.load_state_dict(weights)
        self.modality_mappings = modality_mappings

    def forward(self, x: torch.Tensor, modality: str):
        if x is None: return None
        idx = torch.tensor(self.modality_mappings[modality], dtype=torch.long, device=x.device)
        return self.norm(x + self.modal_embedding(idx).view(1, 1, 1, -1))


class TemporalEncoder(nn.Module, TimeMaskSwitchable):
    def __init__(self, dim, max_length: int, timestep_duration: int, modality: TimeMaskSwitchableProperties,
                 layers: int = 1, heads: int = 8, dropout: float = 0.0):
        nn.Module.__init__(self)
        TimeMaskSwitchable.__init__(self)
        self.enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer=self.enc_layer, num_layers=layers)
        self.pos_embedding = nn.Parameter(torch.randn(1, int(max_length / timestep_duration), dim))  # or sinusoidal
        self.set_attention_modality(modality)  # Initialize the attn modality

    def forward(self, x, mask=None):  # x: (B,T,D), mask: (B,T) bool True=valid
        t = x.size(1)
        x = x + self.pos_embedding[:, :t]

        attn_mask = self._get_attn_mask(t, x.device)
        if mask is None:
            return self.enc(x, mask=attn_mask)

        mask = mask.bool()
        valid = mask.any(dim=1)

        out = x.new_zeros(x.shape)
        if valid.any():
            out[valid] = self.enc(x[valid], src_key_padding_mask=~mask[valid], mask=attn_mask)  # -> (B,T,D)

        return out
