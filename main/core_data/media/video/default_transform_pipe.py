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
from main.dataset.base_config import DatasetConfig


def vid_vivit_interleaved_transform_pipe(config: DatasetConfig) \
        -> tuple[str, nn.Module]:
    return Video.modality_code(), nn.Sequential(
        SubclipVideo(),
        VideoToTensor(),
        ViVitImageProcessorTransform(),
        v2.Lambda(lambda x: x.pixel_values),
        VideoSequenceResampling(
            original_fps=config.vid_source_config.fps,
            sequence_duration_seconds=config.unit_seconds,
            # frames_resampler=RegularFrameResampling(target_config.max_frames, drop_mask=True)
            frames_resampler=RegularFrameResampling(max_length=int(config.unit_seconds), drop_mask=True),
        ),
        ViVitEmbedderTransform(map_to="cpu"),
        ViVitPyramidPatchPooling(),
        MultimediaPadding(max_length=math.ceil(config.max_length / config.unit_seconds)),
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
