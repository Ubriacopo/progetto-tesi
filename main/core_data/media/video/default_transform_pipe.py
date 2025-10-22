import math

import torch
from torch import nn
from torchvision.transforms import v2

from main.core_data.processing.transform import MultimediaPadding, ToSimpleMaskedObject
from main.core_data.media.video import Video
from main.core_data.media.video import VidTargetConfig
from main.core_data.media.video.transforms import SubclipVideo, VideoToTensor, ViVitImageProcessorTransform, \
    VideoSequenceResampling, RegularFrameResampling, ViVitEmbedderTransform, VateVideoResamplerTransform, \
    UnbufferedResize, ViVitForVideoClassificationEmbedderTransform, ViVitPyramidPatchPooling, \
    RecencyBiasedCausalResampling


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
            # frames_resampler=RegularFrameResampling(target_config.max_frames, drop_mask=True)
            frames_resampler=RecencyBiasedCausalResampling(
                max_length=target_config.i_max_length, fps=fps, window_seconds=3, drop_mask=True
            ),
        ),
        ViVitEmbedderTransform(map_to="cpu"),
        ViVitPyramidPatchPooling(),
        MultimediaPadding(max_length=math.ceil(max_length / target_config.i_max_length)),
    )


def vid_vate_basic_transform_pipe(target_config: VidTargetConfig) -> tuple[str, nn.Module]:
    return Video.modality_code(), nn.Sequential(
        SubclipVideo(),
        UnbufferedResize((224, 224)),
        VateVideoResamplerTransform(min_frames=target_config.max_frames),
        v2.Lambda(lambda x: torch.tensor(x)),
        RegularFrameResampling(target_config.max_frames, drop_mask=True),
        ViVitImageProcessorTransform(),
        ViVitForVideoClassificationEmbedderTransform(),
        v2.Lambda(lambda x: x.to("cpu")),
        ToSimpleMaskedObject(stop_at_dim=-1)
    )
