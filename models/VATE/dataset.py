from __future__ import annotations

from torch import nn
from torchvision.transforms import v2

from common.data.audio.transforms import ComputeFeatureHubert, HubertBaseFeatureExtractor
from common.data.data_point import EEGDatasetTransformWrapper
from common.data.video.transforms import ViVitImageProcessorTransform


# Transforms per AMIGOS
def VATE_AMIGOS_transforms() -> EEGDatasetTransformWrapper:
    return EEGDatasetTransformWrapper(
        name="VATE",
        vid_transform=[
            ViVitImageProcessorTransform(),
        ],
        aud_transform=[
            # TODO Vedi se vera la frequenza di amigos e noi salvata cosi
            ComputeFeatureHubert(original_fs=44100),
            # todo vedi se devo gestire in qualche modo medias.
            HubertBaseFeatureExtractor()
        ],
        txt_transform=[
            # Text is for now disabled. I have yet to find a way to extract it correctly.
            v2.Lambda(lambda x: None),
        ],
    )


class FaceCrop(nn.Module):
    # Always replicate the training pipeline end-to-end -> FACE crop on videos to VATE.
    # TODO: Implement (Per ora posticipo in quanto già face videos)
    def forward(self, video):
        return video
