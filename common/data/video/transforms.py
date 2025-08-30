import dataclasses

import torch
import torch.nn.functional as F
import torchcodec
from moviepy import VideoFileClip
from torchcodec.decoders import VideoDecoder
from torchvision.transforms import v2

from common.data.data_point import EEGDatasetDataPoint
from .video import Video



@dataclasses.dataclass
class ResampleFrames:
    fps_map: tuple[int, int]

    def __call__(self, x: list[torch.Tensor], fps: int = None, *args, **kwargs) -> tuple[list[torch.Tensor], dict]:
        if not isinstance(x, list):
            raise TypeError("ResampleFrames only supports list of torch.Tensors")
        if len(x) == 0:
            raise ValueError("Empty frame list cannot be handled")
        if not isinstance(x[0], torch.Tensor):
            raise TypeError("ResampleFrames only supports torch.Tensors not:", type(x[0]))

        x = torch.stack(x, dim=0)
        if x.dim() != 4:
            raise ValueError("Video must be 4D (T,C,H,W) or (T,H,W,C)")

        channels_last = x.shape[-1] in (1, 3)
        x = x.permute(3, 0, 1, 2) if channels_last else x.permute(1, 0, 2, 3)

        c, t, h, w = x.shape

        source_fps = self.fps_map[0] if fps is None else fps
        target_fps = self.fps_map[1]
        new_t = max(1, int(round(t * source_fps / target_fps)))

        out = F.interpolate(x.unsqueeze(0), size=(c, new_t, h, w), mode="bilinear", align_corners=True).squeeze(0)
        out = out.permute(1, 2, 3, 0) if channels_last else out.permute(1, 0, 2, 3)
        out = list(out.unbind(0))

        # Update context information to pass down the line.
        kwargs["fps"] = target_fps
        kwargs["original_fps"] = source_fps
        return out, kwargs


@dataclasses.dataclass
class ResampleVideoDataPoint:
    fps_map: tuple[int, int]

    def __call__(self, x: Video | EEGDatasetDataPoint):
        v = x.vid if isinstance(x, EEGDatasetDataPoint) else x
        d: VideoFileClip = v.data

        if not isinstance(d, VideoFileClip):
            raise TypeError("Inside of Video data we suppose (for the moment) to only have VideoFileClip data")

        v.data = d.with_fps(self.fps_map[1])
        return x


@dataclasses.dataclass
class ResizeVideo:
    new_size: tuple[int, int] | int

    def __call__(self, x: list[torch.Tensor] | EEGDatasetDataPoint | Video):
        if isinstance(x, EEGDatasetDataPoint) or isinstance(x, Video):
            o = x.vid if isinstance(x, EEGDatasetDataPoint) else x

            if o.data is None:
                start, stop = o.interval
                o.data = VideoFileClip(o.file_path).subclipped(start, stop)

            clip: VideoFileClip = o.data
            o.data = clip.resized(height=self.new_size[0])
            o.resolution = o.data.size

            return x

        else:
            return v2.Resize(self.new_size)(x)


class ToVideoFileClip:
    def __call__(self, x: EEGDatasetDataPoint | Video):
        o = x.vid if isinstance(x, EEGDatasetDataPoint) else x

        if o.file_path is None:
            raise ValueError("To transform a Video into VideoFileClip we require a reference file")

        o.data = VideoFileClip(o.file_path)
        return x


class VideoFileToTensor:
    def __call__(self, x: Video) -> list[torch.Tensor]:
        if not isinstance(x, Video):
            raise TypeError("Input object must be of type Video")
        return list([torch.tensor(frame) for frame in x.data.iter_frames()])
