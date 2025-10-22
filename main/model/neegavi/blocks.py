from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from main.model.neegavi.kd import KDHead
from main.utils.data import MaskedValue, KdMaskedValue


class ModalityStream(nn.Module):
    def __init__(self, code: str, output_size: int, adapter: nn.Module,
                 kd_head: KDHead = None, post_kd_adapter: nn.Module = None, time_step_length: float = 1.0):
        super().__init__()

        self.output_size: int = output_size
        self.code: str = code
        self.adapter: nn.Module = adapter

        self.post_kd_adapter: Optional[nn.Module] = post_kd_adapter
        if self.post_kd_adapter is not None and not self.use_kd:
            raise ValueError("You have to use KD to use the post_kd_adapter")

        self.use_kd: bool = kd_head is not None
        self.kd_head: Optional[KDHead] = kd_head
        self.time_step_length: float = time_step_length

    def forward(self, x: torch.Tensor, mask=None, use_kd=True, **kwargs) -> MaskedValue | KdMaskedValue:
        output = {"data": x, "mask": mask}
        y: MaskedValue = self.adapter(x, mask=mask)
        if use_kd and self.use_kd:
            output["kd"] = self.kd_head(y["data"], mask=y["mask"])
        if self.post_kd_adapter is not None:
            y |= self.post_kd_adapter()
        return output | y

    def get_code(self):
        return self.code

    def as_tuple(self) -> tuple[str, ModalityStream]:
        return self.code, self


class SimpleFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int) -> None:
        super().__init__()
        assert mult > 0, "Multiplication has to be a positive integer"
        x, y = dim, dim * mult
        self.net = nn.Sequential(
            nn.LayerNorm(x),  # Normalize
            nn.Linear(x, y, bias=False),  # Map to new shape
            nn.GELU(),  # Non-linearity
            nn.Linear(y, x, bias=False),  # Rebuild the original shape
        )

    def forward(self, x):
        return self.net(x)


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
        return self.norm(x + self.modal_embeddings(idx).view(1, 1, 1, -1))


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
