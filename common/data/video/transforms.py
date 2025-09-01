from dataclasses import replace
from typing import Literal

import torch
from moviepy import VideoFileClip
from torch import nn
from torchcodec.decoders import VideoDecoder

from .video import Video


def check_video_data(x, data_type: type):
    if not isinstance(x, Video):
        raise TypeError("Given object is not of required type Video")

    if x.data is None:
        raise ValueError("Video has to be loaded first.")

    if not isinstance(x.data, data_type):
        raise TypeError("Given video object is not valid")


class VideoToTensor(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    def forward(self, x: Video) -> torch.Tensor:
        frames: torch.Tensor = x.data

        if x.data is None:
            frames = VideoDecoder(x.file_path, device=self.device)[:]
        elif isinstance(x.data, VideoFileClip):
            frames = torch.stack([torch.tensor(frame) for frame in x.data.iter_frames()])

        return frames


class UnbufferedResize(nn.Module):
    def __init__(self, new_size: tuple[int, int] | int):
        super().__init__()
        self.new_size = new_size

    def forward(self, x: Video):
        clip: VideoFileClip = x.data
        check_video_data(x, VideoFileClip)
        return replace(x, data=clip.resized(height=self.new_size[0]), resolution=self.new_size)


class SubclipVideo(nn.Module):
    def forward(self, x: Video):
        check_video_data(x, VideoFileClip)
        return replace(x, data=x.data.subclipped(x.interval[0], x.interval[1]))


class RegularFrameResampling(nn.Module):
    def __init__(self, max_length: int, device="cpu",
                 padding: Literal['zero', 'none'] = 'zero', drop_mask: bool = True):
        super().__init__()
        self.max_length: int = max_length
        self.device = device

        # Possible padding choices
        self.padding: Literal['zero', 'none'] = padding
        self.drop_mask: bool = drop_mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        T, c, h, w = x.shape

        if T > self.max_length:
            i = torch.arange(self.max_length, device=self.device)
            idx = torch.div(i * (T - 1), (self.max_length - 1), rounding_mode="floor").to(torch.long)
            mask = torch.ones(T, dtype=torch.bool, device=x.device)
            return (x[idx], mask) if not self.drop_mask else x[idx]

        if T == self.max_length:
            mask = torch.ones(T, dtype=torch.bool, device=x.device)
            return (x, mask) if not self.drop_mask else x

        if self.padding == "zero":
            # Video is not long enough so we need to pad
            pad = torch.zeros(self.max_length - T, c, h, w)
            # Add the missing frames
            x = torch.cat([x, pad])
            mask = torch.zeros(self.max_length, dtype=torch.bool, device=x.device)
            mask[:T] = True  # We are padding right
            return (x, mask) if not self.drop_mask else x

        if self.padding == "none":
            print("Warning this is plain sequence with 'non' padding rule while required for the current"
                  " sequence. (", str(T), " > ", self.max_length, "). This might cause problems later.")
            return (x, None) if not self.drop_mask else x

        raise NotImplementedError("Given padding modality is invalid and input requires one.")
