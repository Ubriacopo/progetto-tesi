import numpy as np
import torch
import torch.nn.functional as F


class ResampleFps:
    def __init__(self, fps_map: tuple[int, int]):
        """

        :param fps_map: Tuple containing the original fps and the target fps to map the video to.
        """
        self.og_fps, self.new_fps = fps_map

    def __call__(self, video: list[torch.Tensor] | torch.Tensor, **kwargs):
        if isinstance(video, list):
            check_reference = video[0]
            if isinstance(check_reference, np.ndarray):
                video = np.array(video)
                video = torch.Tensor(video)
            elif isinstance(check_reference, torch.Tensor):
                video = torch.stack(video, dim=0)
            else:
                raise TypeError("Given data is not valid")

        if video.dim() != 4:
            raise ValueError("Video must be 4D (T,C,H,W) or (T,H,W,C)")

        channels_last = video.shape[-1] in (1, 3)
        video = video.permute(3, 0, 1, 2) if channels_last else video.permute(1, 0, 2, 3)

        c, t, h, w = video.shape
        new_t = max(1, int(round(t * self.new_fps / self.og_fps)))

        out = F.interpolate(video.unsqueeze(0), size=(new_t, h, w), mode="trilinear", align_corners=False).squeeze(
            0)
        out = out.permute(1, 2, 3, 0) if channels_last else out.permute(1, 0, 2, 3)
        return list(out.unbind(0))
