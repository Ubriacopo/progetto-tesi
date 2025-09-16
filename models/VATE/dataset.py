from __future__ import annotations

from torch import nn

""" 
# Transforms per AMIGOS
def VATE_AMIGOS_transforms() -> EEGDatasetTransformWrapper:
    return EEGDatasetTransformWrapper(
        name="VATE",
        vid_transform=[
            FaceCrop(),
            v2.Lambda(lambda x: x.to("cuda")),
            ViVitImageProcessorTransform(),
            ViVitForVideoClassificationEmbedderTransform(),
        ],
        aud_transform=[
            # TODO Vedi se vera la frequenza di amigos e noi salvata cosi
            ComputeFeatureHubert(original_fs=AmigosConfig.original_aud_fs),
            # todo vedi se devo gestire in qualche modo medias.
            Rearrange("(i D) -> i D", i=1),
            v2.Lambda(lambda x: x.to("cuda")),
            HubertBaseFeatureExtractor()
        ],
        txt_transform=[
            # Text is for now disabled. I have yet to find a way to extract it correctly.
            v2.Lambda(lambda x: None),
        ],
    )
"""


class FaceCrop(nn.Module):
    # Always replicate the training pipeline end-to-end -> FACE crop on videos to VATE.
    # TODO: Implement (Per ora posticipo in quanto già face videos)
    def forward(self, video):
        return video
