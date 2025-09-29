from torch import nn
from torchvision.transforms import v2

from core_data.media.video import Video
from core_data.media.video.transforms import SubclipVideo, VideoToTensor, ViVitImageProcessorTransform, \
    RegularFrameResampling, ViVitEmbedderTransform

# todo copia da repo loro quel che vuole che sia fat
def vate_vid_transform_pipe(max_length: int = 32):
    return Video.modality_code(), nn.Sequential(
        SubclipVideo(),
        VideoToTensor(),
        ViVitImageProcessorTransform(),
        v2.Lambda(lambda x: x.pixel_values),
        # For ViVit masking makes no sense
        RegularFrameResampling(max_length=max_length, drop_mask=True),
        ViVitEmbedderTransform()
    )
