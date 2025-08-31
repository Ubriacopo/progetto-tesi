import torch
from moviepy import VideoFileClip
from torchcodec.decoders import VideoDecoder

from .video import Video
from ..transform import CustomBaseTransform


class VideoToTensor(CustomBaseTransform):
    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device

    @classmethod
    def scriptable(cls) -> bool:
        return False  # Has custom data types. I think this won't allow it to be scriptable

    def do(self, x: Video) -> tuple[torch.Tensor, dict] | None:
        frames: torch.Tensor = x.data
        if x.data is None:
            frames = VideoDecoder(x.file_path, device=self.device)[:]
        elif isinstance(x.data, VideoFileClip):
            frames = torch.stack([torch.tensor(frame) for frame in x.data.iter_frames()])

        metadata = x.to_dict(metadata_only=True)
        return frames, metadata


class RegularFrameResampling(CustomBaseTransform):
    def __init__(self, max_length: int, device="cpu", pad: bool = True):
        super().__init__()
        self.max_length: int = max_length
        self.device = device
        self.pad: bool = pad

    def do(self, x: torch.Tensor) -> tuple[torch.Tensor, dict] | None:
        T, c, h, w = x.shape
        metadata = {}
        if T > self.max_length:
            i = torch.arange(self.max_length, device=self.device)
            idx = torch.div(i * (T - 1), (self.max_length - 1), rounding_mode="floor").to(torch.long)

            metadata["frames_indexes"] = idx
            return x[idx], metadata

        if self.pad:
            # Video is not long enough so we need to pad
            pad = torch.zeros(self.max_length - T, c, h, w)
            # Add the missing frames
            x = torch.cat([x, pad])
            metadata["mask"] = torch.cat([torch.ones(T), pad], dim=0)
            return x, metadata

        print("Warning this is plain sequence without padding. It might break later.")
        return x, metadata
