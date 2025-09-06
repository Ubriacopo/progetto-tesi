import dataclasses

from moviepy import VideoFileClip
from torch import nn

from common.data.audio.transforms import SubclipAudio
from common.data.data_point import EEGDatasetDataPoint
from common.data.video.transforms import SubclipVideo


@dataclasses.dataclass
class ResizeEEGDataPointMedia(nn.Module):
    def __init__(self, new_size: tuple[int, int] | int):
        super().__init__()
        self.new_size = new_size

    def forward(self, x: EEGDatasetDataPoint):
        clip: VideoFileClip = x.vid.data
        if clip is None:
            # Make the clip if this is the first clipping experience.
            start, stop = x.vid.interval
            clip = VideoFileClip(x.vid.file_path).subclipped(start, stop)

        # Update our data
        x.vid.resolution = self.new_size
        x.vid.data = clip.resized(height=self.new_size[0])

        return x


class SubclipMedia(nn.Module):
    def forward(self, x: EEGDatasetDataPoint):
        x.aud = SubclipAudio()(x.aud)
        x.vid = SubclipVideo()(x.vid)
        return x
