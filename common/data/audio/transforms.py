from typing import Optional

import torch
from moviepy import AudioFileClip

from .audio import Audio


class AudioToTensor:
    def __call__(self, x: Audio, *args, **kwargs):
        aud: Optional[AudioFileClip] = x.data
        metadata = x.to_dict(metadata_only=True)

        if x.data is None:
            aud = AudioFileClip(x.file_path)

        x = aud.to_soundarray()
        x = torch.from_numpy(x).float()
        x = x.T
        return x, kwargs | metadata


class ToMono:
    """
    Transforms a source from Stereo or any other format to MONO. (Single wave)
    """

    def __call__(self, x: torch.Tensor, **kwargs):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected a torch.Tensor")

        return torch.mean(x, dim=0, keepdim=True)
