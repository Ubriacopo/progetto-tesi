from typing import TypedDict, Optional

import torch
from torch import nn

from main.core_data.media.audio import Audio
from main.core_data.media.text import Text
from main.core_data.media.video import Video
from main.utils.data import MaskedValue


def build_sequential(input_size, hidden_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


class MaskedContrastiveModelOutputs(TypedDict):
    vid: MaskedValue
    aud: MaskedValue
    txt: MaskedValue


class MaskedContrastiveModel(nn.Module):
    keys = [Video.modality_code(), Audio.modality_code(), Text.modality_code()]

    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.hidden_channels: int = hidden_channels
        self.out_channels: int = out_channels
        self.embedding_video = build_sequential(400, self.hidden_channels, self.out_channels)
        self.embedding_audio = build_sequential(768, self.hidden_channels, self.out_channels)
        self.embedding_text = build_sequential(768, self.hidden_channels, self.out_channels)
        logit_scale_init_value = 2.6592
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))

    def _process_modality(self, o: Optional[MaskedValue], embedder: nn.Module, device, b):
        if o is not None:
            emb = nn.functional.normalize(embedder(o["data"]), dim=-1)
            emb = emb * o["mask"].bool().unsqueeze(-1)
            return MaskedValue(data=emb, mask=o["mask"].bool())

        empty = torch.zeros(b, self.out_channels, device=device)
        return MaskedValue(data=empty, mask=torch.zeros(b, device=device, dtype=torch.bool))

    def forward(self, x: dict, **kwargs) \
            -> MaskedContrastiveModelOutputs:
        first = next(iter(x.values()))
        device, b = first["data"].device, first["data"].shape[0]
        return {
            "vid": self._process_modality(x.get(Video.modality_code(), None), self.embedding_video, device, b),
            "aud": self._process_modality(x.get(Audio.modality_code(), None), self.embedding_audio, device, b),
            "txt": self._process_modality(x.get(Text.modality_code(), None), self.embedding_text, device, b),
        }
