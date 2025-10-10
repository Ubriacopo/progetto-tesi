from typing import TypedDict

import torch
from torch import nn

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
    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.hidden_channels: int = hidden_channels
        self.out_channels: int = out_channels
        self.embedding_video = build_sequential(400, self.hidden_channels, self.out_channels)
        self.embedding_audio = build_sequential(768, self.hidden_channels, self.out_channels)
        self.embedding_text = build_sequential(768, self.hidden_channels, self.out_channels)
        logit_scale_init_value = 2.6592
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))

    def forward(self, vid: MaskedValue, aud: MaskedValue, txt: MaskedValue) \
            -> MaskedContrastiveModelOutputs:
        vid_data, vid_mask = vid["data"], vid["mask"]
        vid_data = self.embedding_video(vid_data)
        vid_data = nn.functional.normalize(vid_data)

        aud_data, aud_mask = aud["data"], aud["mask"]
        if aud_mask.any():
            aud_data = self.embedding_audio(aud_data)
            aud_data = nn.functional.normalize(aud_data)

        txt_data, txt_mask = txt["data"], txt["mask"]
        if txt_mask.any():
            txt_data = self.embedding_text(txt_data)
            txt_data = nn.functional.normalize(txt_data)

        return {
            "vid": MaskedValue(data=vid_data, mask=vid_mask),
            "aud": MaskedValue(data=aud_data, mask=aud_mask),
            "txt": MaskedValue(data=txt_data, mask=txt_mask)
        }
