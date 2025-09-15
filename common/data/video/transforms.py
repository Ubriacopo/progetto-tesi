from dataclasses import replace
from typing import Literal, Optional

import torch
from moviepy import VideoFileClip
from torch import nn, dtype
from torchcodec.decoders import VideoDecoder
from transformers import VivitImageProcessor, VivitForVideoClassification

from .video import Video


def check_video_data(x, data_type: type):
    if not isinstance(x, Video):
        raise TypeError("Given object is not of required type Video")

    if x.data is None:
        raise ValueError("Video has to be loaded first.")

    if not isinstance(x.data, data_type):
        raise TypeError("Given video object is not valid")


class VideoToTensor(nn.Module):
    def __init__(self, device="cpu", tensor_dtype: dtype = torch.float32):
        super().__init__()
        self.device = device
        self.tensor_dtype = tensor_dtype

    def forward(self, x: Video) -> torch.Tensor:
        frames: torch.Tensor = x.data

        if x.data is None:
            frames = VideoDecoder(x.file_path, device=self.device)[:]
        elif isinstance(x.data, VideoFileClip):
            frames = torch.stack([torch.tensor(frame) for frame in x.data.iter_frames()])
        return frames.type(dtype=self.tensor_dtype)


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


class VideoSequenceResampling(nn.Module):
    def __init__(self, original_fps: int, sequence_duration_seconds: int, frames_resampler: nn.Module):
        super().__init__()
        self.sequence_length = original_fps * sequence_duration_seconds
        self.frames_resampler = frames_resampler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, c, h, w = x.shape
        segments = int(T / self.sequence_length)

        if T % self.sequence_length != 0:
            segments += 1

        points = x.unbind(0)
        y: Optional[torch.Tensor] = None
        for i in range(segments):
            segment_points = points[i * self.sequence_length:(i + 1) * self.sequence_length]
            res = self.frames_resampler(torch.stack(segment_points))
            res = res.unsqueeze(0)  # We have new dimension that records the sequence.
            y: torch.Tensor = torch.cat((y, res)) if y is not None else res
        return y


class RegularFrameResampling(nn.Module):
    def __init__(self, max_length: int, device="cpu",
                 padding: Literal['zero', 'last', 'none'] = 'zero', drop_mask: bool = True):
        super().__init__()
        self.max_length: int = max_length
        self.device = device

        # Possible padding choices
        self.padding: Literal['zero', 'last', 'none'] = padding
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

        if self.padding == "last":
            pad = x[-1].repeat(self.max_length - T)
            x = torch.cat([x, pad])
            mask = torch.zeros(self.max_length, dtype=torch.bool, device=x.device)
            mask[:T] = True  # We are padding right
            return (x, mask) if not self.drop_mask else x

        if self.padding == "none":
            print("Warning this is plain sequence with 'non' padding rule while required for the current"
                  " sequence. (", str(T), " > ", self.max_length, "). This might cause problems later.")
            return (x, None) if not self.drop_mask else x

        raise NotImplementedError("Given padding modality is invalid and input requires one.")


# todo da metter in embedder zone
# todo visionare bene con time sequences.
class ViVitImageProcessorTransform(nn.Module):
    def __init__(self, model_name: str = "google/vivit-b-16x2-kinetics400",
                 processor: VivitImageProcessor = None, force_time_seq: bool = False):
        super().__init__()

        self.processor: VivitImageProcessor = processor
        if processor is None:
            self.processor: VivitImageProcessor = VivitImageProcessor.from_pretrained(model_name)

        self.force_time_seq = force_time_seq

    def forward(self, x):
        if isinstance(x, torch.Tensor) and len(x.shape) == 3:
            x = [x]
        elif isinstance(x, torch.Tensor):
            x = list(x.unbind(0))

        x = self.processor.preprocess(x, return_tensors="pt")
        if not self.force_time_seq:
            x["pixel_values"] = x["pixel_values"].squeeze(0)

        return x


class ViVitFeatureExtractorTransform(nn.Module):
    def __init__(self, model_name: str = "google/vivit-b-16x2-kinetics400", force_time_seq: bool = False, device=None):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = VivitForVideoClassification.from_pretrained(model_name, device_map=device)
        self.force_time_seq = force_time_seq

    def forward(self, x):
        with torch.no_grad():
            x["pixel_values"] = x["pixel_values"].unsqueeze(0)
            x = x.to(self.model.device)  # In case they differ!
            x = self.model(**x).logits.squeeze(0)

        return x
