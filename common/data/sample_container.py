import dataclasses
from typing import Optional, Text

from moviepy import VideoFileClip, AudioFileClip

from common.data.audio import Audio
from common.data.eeg.eeg import EEG
from common.data.loader import EEGDatasetDataPoint
from common.data.video.video import Video


@dataclasses.dataclass
class EEGDatasetSampleContainer:
    entry_id: str  # We suppose every entry has a unique way of identifying itself
    # EEG dataset supposes to have for an entry a list of recordings. Other data types are optional.
    eeg: EEG
    vid: Optional[Video] = None
    txt: Optional[Text] = None
    aud: Optional[Audio] = None


# todo finire con le transform (a blocchetti).
class ResizeVideo:
    def __init__(self, new_size: tuple[int, int] | int):
        self.new_size = new_size

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

        if vid is None:
            # Make the clip if this is the first clipping experience.
            vid = VideoFileClip(x.vid.file_path)
            aud = vid.audio

        start, stop = x.vid.interval
        x.vid.data = vid.subclipped(start, stop)
        x.aud.data = aud.subclipped(start, stop)

        return x
