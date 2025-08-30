import dataclasses

import numpy as np
import torch
import torch.nn.functional as F
from moviepy import VideoFileClip
from torchvision.transforms import v2

from common.data.data_point import EEGDatasetDataPoint
from .video import Video


@dataclasses.dataclass
class ResampleVideoFrames:
    fps_map: tuple[int, int]

    def __call__(self, x: list[torch.Tensor] | Video | EEGDatasetDataPoint):
        o = x if not isinstance(x, EEGDatasetDataPoint) else x.vid
        if isinstance(o, Video) and isinstance(o.data, VideoFileClip):
            ResampleVideoDataPoint(self.fps_map)(o)
            return x

        if isinstance(o, Video):
            o.data = ResampleVideoTensor(o.data)
            return x

        # Tensor Object path
        return ResampleVideoTensor(self.fps_map)(x)


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
class ResampleVideoTensor:
    fps_map: tuple[int, int]

    def __call__(self, x: list[torch.Tensor] | torch.Tensor, **kwargs):
        if isinstance(x, list):
            check_reference = x[0]
            if isinstance(check_reference, np.ndarray):
                # Turn list into np array to tensor (freezes if I do directly with torch).
                x = torch.Tensor(np.array(x))
            elif isinstance(check_reference, torch.Tensor):
                x = torch.stack(x, dim=0)
            else:
                raise TypeError("Given data is not valid")

        if x.dim() != 4:
            raise ValueError("Video must be 4D (T,C,H,W) or (T,H,W,C)")

        channels_last = x.shape[-1] in (1, 3)
        x = x.permute(3, 0, 1, 2) if channels_last else x.permute(1, 0, 2, 3)

        c, t, h, w = x.shape
        new_t = max(1, int(round(t * self.fps_map[0] / self.fps_map[1])))

        out = F.interpolate(x.unsqueeze(0), size=(new_t, h, w), mode="trilinear", align_corners=False).squeeze(0)
        out = out.permute(1, 2, 3, 0) if channels_last else out.permute(1, 0, 2, 3)
        out = list(out.unbind(0))

        return out


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
