from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Optional, Literal

import torch
from torch import nn

from main.model.neegavi.kd import KDHead
from main.utils.data import MaskedValue, KdMaskedValue


@dataclasses.dataclass
class TimeMaskSwitchableProperties:
    mode: Literal['causal', 'bidirectional', 'window']
    # Note: For now I avoid using the window. Might be improvement. TODO: Valuta
    lookback: Optional[int] = None  # Size of the window if in window mode
    lookahead: int = 0  # Future tokens allowed (This makes window on past only possible).


class TimeMaskSwitchable(ABC):
    def __init__(self):
        """
        A time mask switchable model changes how it operates the attention mask over time steps.

        Modalities are:
        - bidirectional: Both past and future can be seen
        - causal: Only past and the current step can be seen. So for step i only k with k < i

        Both can apply a windowing by checking lookahead and lookback.
        For the time being windowing is not used.
        """
        super().__init__()
        self.modality: Optional[TimeMaskSwitchableProperties] = None
        self.modality_cache: dict = {}

    def set_attention_modality(self, modality: TimeMaskSwitchableProperties) -> None:
        self.modality = modality

    def _get_attn_mask(self, t: int, device):
        if self.modality.mode == "bidirectional":
            return None  # Everything is allowed.

        key = (t, self.modality.mode, self.modality.lookback, self.modality.lookahead, device)
        if key in self.modality_cache:
            # Value already calculated so we return it.
            return self.modality_cache[key]

        if self.modality.mode == "causal":
            mask = torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)

        elif self.modality.mode == "window":
            # Attend to: [t- lookback, t + lookahead]
            i, j = torch.arange(t, device=device), torch.arange(t, device=device)
            lookback, lookahead = self.modality.lookback or 0, self.modality.lookahead
            mask = (j[None, :] < (i[:, None] - lookback)) | (j[None, :] > (i[:, None] + lookahead))

        else:
            raise ValueError(f"Set modality: {self.modality} is invalid")

        self.modality_cache[key] = mask
        return mask


class ModalityStream(nn.Module):
    def __init__(self, code: str, output_size: int, adapter: nn.Module, timestep_seconds: int,
                 kd_head: KDHead = None, post_kd_adapter: nn.Module = None):
        super().__init__()

        self.output_size: int = output_size
        self.code: str = code
        self.adapter: nn.Module = adapter

        self.post_kd_adapter: Optional[nn.Module] = post_kd_adapter
        if self.post_kd_adapter is not None and not self.use_kd:
            raise ValueError("You have to use KD to use the post_kd_adapter")

        self.use_kd: bool = kd_head is not None
        self.kd_head: Optional[KDHead] = kd_head
        self.timestep_seconds: int = timestep_seconds

    def forward(self, x: torch.Tensor, mask=None, use_kd=True, **kwargs) -> MaskedValue | KdMaskedValue:
        if mask is not None and isinstance(mask, torch.Tensor):
            mask = mask.bool()

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
        """
        Two layered feed forward network with GELU activation
        :param dim: Latent space dimension
        :param mult: Feed-forward multiplier
        """
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


class MaskedFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float):
        super().__init__()
        self.net = SimpleFeedForward(dim, mult)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask):
        # x [b, T, p, D]
        # m [b, T]
        m = mask[:, :, None, None].to(x.dtype)
        xm = x * m
        y = xm + self.dropout(self.net(xm))
        return MaskedValue(data=y * m, mask=mask)


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


class TemporalEncoder(nn.Module, TimeMaskSwitchable):
    def __init__(self, dim, max_length: int, timestep_duration: int, modality: TimeMaskSwitchableProperties,
                 layers: int = 1, heads: int = 8, dropout: float = 0.0):
        nn.Module.__init__(self)
        TimeMaskSwitchable.__init__(self)
        self.enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer=self.enc_layer, num_layers=layers)
        self.pos = nn.Parameter(torch.randn(1, int(max_length / timestep_duration), dim))  # or sinusoidal
        self.set_attention_modality(modality)  # Initialize the attn modality

    def forward(self, x, mask=None):  # x: (B,T,D), mask: (B,T) bool True=valid
        t = x.size(1)
        x = x + self.pos[:, :t]

        attn_mask = self._get_attn_mask(t, x.device)
        if mask is None:
            return self.enc(x, mask=attn_mask)

        mask = mask.bool()
        valid = mask.any(dim=1)

        out = x.new_zeros(x.shape)
        if valid.any():
            out[valid] = self.enc(x[valid], src_key_padding_mask=~mask[valid], mask=attn_mask)  # -> (B,T,D)

        return out


# todo vedi se fa a caso nostro
class SlotMLPExpander(nn.Module):
    def __init__(self, dim: int, p: int, hidden_mult: int = 4):
        super().__init__()
        self.p: int = p
        self.slots = nn.Parameter(torch.randn(p, dim) * 0.02)
        h = hidden_mult * dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * dim, h),
            nn.GELU(),
            nn.Linear(h, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        # x: (B,T,D)
        B, T, D = x.shape
        x = self.norm(x)

        s = self.slots[None, None, :, :].expand(B, T, self.p, D)  # (B,T,P,D)
        xr = x[:, :, None, :].expand(B, T, self.p, D)  # (B,T,P,D)
        y = self.mlp(torch.cat([xr, s], dim=-1))  # (B,T,P,D)

        mask_p = mask[:, :, None].expand(B, T, self.p) if mask is not None else None
        return y, mask_p


class AbstractAttentionBlock(nn.Module, ABC):
    @abstractmethod
    def forward(self, q: torch.Tensor, kv: torch.Tensor, attn_mask=None, q_mask=None, kv_mask=None):
        pass
