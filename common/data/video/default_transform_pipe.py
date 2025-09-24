import math

from torch import nn
from torchvision.transforms import v2

from common.data.transform import MultimediaPadding
from common.data.video import Video
from common.data.video.config import VidTargetConfig
from common.data.video.transforms import SubclipVideo, VideoToTensor, ViVitImageProcessorTransform, \
    VideoSequenceResampling, RegularFrameResampling, ViVitEmbedderTransform


def vid_vivit_interleaved_transform_pipe(target_config: VidTargetConfig, fps: int, max_length: int) \
        -> tuple[str, nn.Module]:
    return Video.modality_code(), nn.Sequential(
        SubclipVideo(),
        VideoToTensor(),
        ViVitImageProcessorTransform(),
        v2.Lambda(lambda x: x.pixel_values),
        VideoSequenceResampling(
            original_fps=fps,
            sequence_duration_seconds=target_config.i_max_length,
            frames_resampler=RegularFrameResampling(target_config.max_fps, drop_mask=True)
        ),
        ViVitEmbedderTransform(map_to="cpu"),
        MultimediaPadding(max_length=math.ceil(max_length / target_config.i_max_length)),
    )


def vid_vivit_default_transform_pipe(target_config: VidTargetConfig, fps: int, max_length: int) \
        -> tuple[str, nn.Module]:
    return Video.modality_code(), nn.Sequential(
        SubclipVideo(),
        VideoToTensor(),
        ViVitImageProcessorTransform(),
        v2.Lambda(lambda x: x.pixel_values),
        # For ViVit masking makes no sense
        RegularFrameResampling(max_length=max_length, drop_mask=True),
        ViVitEmbedderTransform()
    )
