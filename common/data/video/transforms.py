import dataclasses

import numpy as np
import torch
import torch.nn.functional as F
from moviepy import VideoFileClip

from common.data.loader import EEGDatasetDataPoint
from common.data.video import Video


@dataclasses.dataclass
class SampleVideoFrames:
    fps_map: tuple[int, int]

    def __call__(self, x: list[torch.Tensor] | Video | EEGDatasetDataPoint):
        if isinstance(x, EEGDatasetDataPoint) or isinstance(x, Video):
            return SampleVideoDataPoint(self.fps_map)(x)
        else:  # Tensor Object path
            return SampleVideoTensor(self.fps_map)(x)


@dataclasses.dataclass
class SampleVideoDataPoint:
    fps_map: tuple[int, int]

    def __call__(self, x: Video | EEGDatasetDataPoint):
        v = x.vid if isinstance(x, EEGDatasetDataPoint) else x
        d: VideoFileClip = v.data
        assert isinstance(d, VideoFileClip), \
            "Inside of Video data we suppose (for the moment) to only have VideoFileClip data"
        v.data = d.with_fps(self.fps_map[1])

        return x


@dataclasses.dataclass
class SampleVideoTensor:
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
