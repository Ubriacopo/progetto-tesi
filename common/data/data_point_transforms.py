import dataclasses

from moviepy import VideoFileClip, AudioFileClip

from common.data.data_point import EEGDatasetDataPoint


@dataclasses.dataclass
class ResizeEEGDataPointMedia:
    new_size: tuple[int, int] | int

    def __call__(self, x: EEGDatasetDataPoint):
        clip: VideoFileClip = x.vid.data
        if clip is None:
            # Make the clip if this is the first clipping experience.
            start, stop = x.vid.interval
            clip = VideoFileClip(x.vid.file_path).subclipped(start, stop)

        # Update our data
        x.vid.resolution = self.new_size
        x.vid.data = clip.resized(height=self.new_size[0])

        return x


class SubclipMedia:
    def __call__(self, x: EEGDatasetDataPoint):
        vid: VideoFileClip = x.vid.data
        aud: AudioFileClip = x.aud.data
        start, stop = x.vid.interval

        if vid is None:
            # Make the clip if this is the first clipping experience.
            vid = VideoFileClip(x.vid.file_path)
            aud = vid.audio

        x.vid.data = vid.subclipped(start, stop)
        x.aud.data = aud.subclipped(start, stop)
        return x
