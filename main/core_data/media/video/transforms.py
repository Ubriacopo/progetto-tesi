import logging
from dataclasses import replace
from typing import Literal, Optional, Iterable

import torch
from einops import rearrange
from moviepy import VideoFileClip
from torch import nn, dtype
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitModel

from main.core_data.utils import timed
from main.core_data.media.video.video_processor import VideoResampler
from main.utils.pyramid_pooling import temporal_pyramid_pooling_3d
from .utils import check_video_data
from .video import Video


class VideoToTensor(nn.Module):
    def __init__(self, device="cpu", tensor_dtype: dtype = torch.float32):
        super().__init__()
        self.device = device
        self.tensor_dtype = tensor_dtype

    def forward(self, x: Video) -> torch.Tensor:
        frames: torch.Tensor = x.data
        if isinstance(x.data, VideoFileClip):
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
    # noinspection PyMethodMayBeStatic
    def forward(self, x: Video):
        check_video_data(x, VideoFileClip)
        return replace(x, data=x.data.subclipped(x.interval[0], x.interval[1]))


class VideoSequenceResampling(nn.Module):
    def __init__(self, original_fps: int, sequence_duration_seconds: int | float, frames_resampler: nn.Module):
        super().__init__()
        self.sequence_length = original_fps * sequence_duration_seconds
        self.frames_resampler = frames_resampler

    @timed()
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
                 padding: Literal['zero', 'last', 'none'] = 'last', drop_mask: bool = True):
        super().__init__()
        self.max_length: int = max_length
        self.device = device

        # Possible padding choices
        self.padding: Literal['zero', 'last', 'none'] = padding
        self.drop_mask: bool = drop_mask

    @timed()
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


class RecencyBiasedCausalResampling(nn.Module):
    """
    Select exactly `max_length` past frames (≤ t) from the last `window_seconds`,
    biased toward recent frames. Deterministic via quantiles of an exponential CDF.
    """
    def __init__(self, max_length: int, fps: int, window_seconds: float,
                 alpha: float = 0.7, padding: Literal['zero', 'last', 'none'] = 'last',
                 device: str = "cpu", drop_mask: bool = True):
        super().__init__()
        self.max_length = max_length
        self.fps = fps
        self.window_seconds = window_seconds
        self.alpha = alpha          # larger → stronger recent bias
        self.padding = padding
        self.device = device
        self.drop_mask = drop_mask

    def _recency_indices(self, T: int) -> torch.Tensor:
        # Use only the last N frames (causal window)
        N = min(T, int(round(self.window_seconds * self.fps)))
        start = T - N  # inclusive
        # distances from "now" (0 = most recent)
        d = torch.arange(N-1, -1, -1, device=self.device, dtype=torch.float32)  # [N]
        # exponential recency weights
        w = torch.exp(-self.alpha * (d / max(self.fps, 1)))
        cdf = torch.cumsum(w, dim=0); cdf = cdf / cdf[-1]
        # deterministic quantile picks (denser near recent frames)
        q = (torch.arange(self.max_length, device=self.device, dtype=torch.float32) + 0.5) / self.max_length
        idx_rel = torch.searchsorted(cdf, q).clamp(max=N-1)          # [max_length], 0..N-1
        idx_abs = start + (N-1 - idx_rel)
        idx_abs = start + idx_rel  # map back to absolute, ascending
        # ensure strictly non-decreasing & length = max_length
        return idx_abs.to(torch.long)

    @timed()
    def forward(self, x: torch.Tensor):
        """
        x: [T, C, H, W] where x[-1] is the most recent frame at time t.
        Returns: [max_length, C, H, W] (and optional mask if drop_mask=False).
        """
        T, C, H, W = x.shape

        if T >= self.max_length:
            idx = self._recency_indices(T)
            out = x.index_select(0, idx)
            if self.drop_mask: return out
            mask = torch.ones(self.max_length, dtype=torch.bool, device=x.device)
            return out, mask

        # T < max_length → pad on the right (future) with zeros or last frame (still causal)
        if self.padding == "zero":
            pad = torch.zeros(self.max_length - T, C, H, W, device=x.device, dtype=x.dtype)
        elif self.padding == "last":
            pad = x[-1:].expand(self.max_length - T, -1, -1, -1).contiguous()
        elif self.padding == "none":
            if self.drop_mask: return x
            return x, None
        else:
            raise NotImplementedError("Invalid padding mode.")

        out = torch.cat([x, pad], dim=0)
        if self.drop_mask: return out
        mask = torch.zeros(self.max_length, dtype=torch.bool, device=x.device); mask[:T] = True
        return out, mask



class ViVitImageProcessorTransform(nn.Module):
    def __init__(self, model_name: str = "google/vivit-b-16x2-kinetics400",
                 processor: VivitImageProcessor = None, force_time_seq: bool = False):
        super().__init__()

        self.processor: VivitImageProcessor = processor
        if processor is None:
            self.processor: VivitImageProcessor = VivitImageProcessor.from_pretrained(model_name)

        self.force_time_seq = force_time_seq

    @torch.inference_mode()
    @timed()
    def forward(self, x):
        if isinstance(x, torch.Tensor) and len(x.shape) == 3:
            x = [x]
        elif isinstance(x, torch.Tensor):
            x = list(x.unbind(0))

        x = self.processor.preprocess(x, return_tensors="pt")
        if not self.force_time_seq:
            x["pixel_values"] = x["pixel_values"].squeeze(0)

        return x


class ViVitEmbedderTransform(nn.Module):
    def __init__(self, model_name: str = "google/vivit-b-16x2-kinetics400", device=None,
                 map_to=None, mini_batch_size: int = None):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = VivitModel.from_pretrained(model_name, device_map=device)

        self.device = device
        # If the device is the same we don't have to remap.
        self.map_to = map_to if map_to is not None and map_to != self.device else None

        self.mini_batch_size: int = mini_batch_size

    @timed()
    def forward(self, x) -> torch.Tensor:
        if len(x.shape) == 4:
            # Add a virtual batch
            x = x.unsqueeze(0)

        with torch.inference_mode():
            y = self.model(x.to(self.device))

        y = y.last_hidden_state
        # Discard [CLS] token
        tokens = y[:, 1:, :]
        # Move to CPU if wanted
        if self.map_to is not None:
            tokens = tokens.to(self.map_to)
        return tokens


class ViVitForVideoClassificationEmbedderTransform(nn.Module):
    def __init__(self, model_name: str = "google/vivit-b-16x2-kinetics400", force_time_seq: bool = False, device=None):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = VivitForVideoClassification.from_pretrained(model_name, device_map=device)
        self.force_time_seq = force_time_seq

    @timed()
    def forward(self, x):
        with torch.inference_mode():
            x["pixel_values"] = x["pixel_values"].unsqueeze(0)
            x = x.to(self.model.device)  # In case they differ!
            x = self.model(**x).logits.squeeze(0)

        return x


class VateVideoResamplerTransform(nn.Module):
    def __init__(self, min_frames: int, detect_conf: float = 0.5, reduce_bbox: float = 0.1):
        super().__init__()
        self.video_resampler = VideoResampler(detect_conf=detect_conf, reduce_bbox=reduce_bbox, min_frames=min_frames)

    @timed()
    def forward(self, x: Video) -> torch.Tensor:
        x.data = torch.tensor(self.video_resampler.resample_clip(x.data))
        return x.data


class ViVitPyramidPatchPooling(nn.Module):
    def __init__(self, levels: Iterable[int] = (1, 2, 4, 8, 16, 33)):
        super().__init__()
        self.levels: Iterable[int] = levels
        self.use_pyramid_pooling: bool = True
        if sum(self.levels) >= 16:
            # I will avoid Pyramid pooling
            logging.warning("Pyramid part of pooling is kinda useless if we keep the same patch size."
                            "It might be better to just not do it, thus pyramid pooling is disabled for this iteration.")
            self.use_pyramid_pooling = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "t (P F) D -> t P F D", P=16)  # (Temporal Patch x Frame) decomposition
        # Average pooling over the spatial grid
        x = x.mean(dim=-2)
        # Do pyramid pooling over the temporal tokens
        if self.use_pyramid_pooling:
            x = temporal_pyramid_pooling_3d(x, self.levels)
        return x
