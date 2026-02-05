from __future__ import annotations

import dataclasses
from abc import ABC
from typing import Literal, Optional

import torch


@dataclasses.dataclass
class TimeMaskSwitchableProperties:
    mode: Literal['causal', 'bidirectional', 'window']
    # Note: For now I avoid using the window. Might be improvement. TODO: Valuta
    lookback: Optional[int] = None  # Size of the window if in window mode
    lookahead: int = 0  # Future tokens allowed (This makes window on past only possible).


class TimeMaskSwitchable(ABC):
    def __init__(self):
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
