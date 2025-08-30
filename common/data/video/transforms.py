import dataclasses

import torch
from moviepy import VideoFileClip
from torchcodec.decoders import VideoDecoder

from .video import Video


class VideoToTensor:
    def __call__(self, x: Video, device: str = "cpu", *args, **kwargs) -> tuple[list[torch.Tensor], dict] | None:
        frames: list[torch.Tensor] = x.data
        if x.data is None:
            decoder = VideoDecoder(x.file_path, device=device)
            frames = list(decoder[:].unbind(0))
        elif isinstance(x.data, VideoFileClip):
            frames = list([torch.tensor(frame) for frame in x.data.iter_frames()])

        metadata = x.to_dict(metadata_only=True)
        return frames, kwargs | metadata


@dataclasses.dataclass
class RegularFrameResampling:
    max_length: int  # Time interval we want
    pad: bool = True
    device = "cpu"

    def __call__(self, x: list[torch.Tensor], fps: int, *args, **kwargs) -> tuple[list[torch.Tensor], dict]:
        if fps is None:
            raise ValueError("Original fps can't be None")

        T = len(x)  # Original sequence length (Frames count)
        c, h, w = x[0].shape

        if T > self.max_length:
            i = torch.arange(self.max_length, device=self.device)
            idx = torch.div(i * (T - 1), (self.max_length - 1), rounding_mode="floor").to(torch.long)

            x = torch.stack(x, 0)
            x = list(x[idx].unbind(0))
            kwargs["frames_indexes"] = idx
            return x, kwargs

        if self.pad:
            # Video is not long enough so we need to pad
            pad = torch.zeros(self.max_length - T, c, h, w)
            # Add the missing frames
            x += list(pad.unbind(0))
            kwargs["mask"] = torch.cat([torch.ones(T), pad], dim=0)
            return x, kwargs

        print("Warning this is plain sequence without padding. It might break later.")
        return x, kwargs
