from typing import Optional

import torch
from moviepy import AudioFileClip
from torchaudio.transforms import Resample

from .audio import Audio
from ..transform import CustomBaseTransform


class AudioToTensor(CustomBaseTransform):
    @classmethod
    def scriptable(cls) -> bool:
        return False  # Has custom data types. I think this won't allow it to be scriptable

    def do(self, x: Audio):
        aud: Optional[AudioFileClip] = x.data
        metadata = x.to_dict(metadata_only=True)

        if x.data is None:
            aud = AudioFileClip(x.file_path)

        x = aud.to_soundarray()
        x = torch.from_numpy(x).float()
        x = x.T
        return x, metadata


class ToMono(CustomBaseTransform):
    """
    Transforms a source from Stereo or any other format to MONO. (Single wave)
    """

    def do(self, x: torch.Tensor, **kwargs):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected a torch.Tensor")
        return torch.mean(x, dim=0, keepdim=True)


# TODO: Evitiamo questa "multi" gestione di multi output. Noi prendiamo solo argomenti definit e fine.
#       Facciamo classic style. TODO
# todo meh ora tutte da ridefinire cosi? Mi sembra assurdo
class MyResample(Resample, CustomBaseTransform):
    def forward(self, x):
        return CustomBaseTransform.forward(self, x)

    def do(self, x):
        return Resample.forward(self, x)
