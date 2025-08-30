import dataclasses

import torch
from moviepy import VideoFileClip
from torchcodec.decoders import VideoDecoder

from .video import Video


class VideoToTensor:
    def __call__(self, x: Video, device: str = "cpu", *args, **kwargs) -> tuple[torch.Tensor, dict] | None:
        frames: torch.Tensor = x.data
        if x.data is None:
            frames = VideoDecoder(x.file_path, device=device)[:]
        elif isinstance(x.data, VideoFileClip):
            frames = torch.stack([torch.tensor(frame) for frame in x.data.iter_frames()])

        metadata = x.to_dict(metadata_only=True)
        return frames, kwargs | metadata


@dataclasses.dataclass
class RegularFrameResampling:
    max_length: int  # Time interval we want
    pad: bool = True
    device = "cpu"

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> tuple[torch.Tensor, dict]:
        T, c, h, w = x.shape

        if T > self.max_length:
            i = torch.arange(self.max_length, device=self.device)
            idx = torch.div(i * (T - 1), (self.max_length - 1), rounding_mode="floor").to(torch.long)

            kwargs["frames_indexes"] = idx
            return x[idx], kwargs

        if self.pad:
            # Video is not long enough so we need to pad
            pad = torch.zeros(self.max_length - T, c, h, w)
            # Add the missing frames
            x = torch.cat([x, pad])
            kwargs["mask"] = torch.cat([torch.ones(T), pad], dim=0)
            return x, kwargs

        print("Warning this is plain sequence without padding. It might break later.")
        return x, kwargs
