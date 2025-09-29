import math

from torch import nn

from core_data.media.text import Text
from core_data.media.text import TxtTargetConfig
from core_data.media.text.transforms import WhisperClipTextExtract, SubclipTextExtract, MiniLMEmbedderTransform
from core_data.processing.transform import MultimediaPadding


def shared_txt_transform_pipe(text_config: TxtTargetConfig, ):
    return Text.modality_code(), nn.Sequential(
        WhisperClipTextExtract(device="cpu"),  # Extracts all texts
    )


def txt_from_aud_interleaved_txt_extract_transform_pipe(target_txt_config: TxtTargetConfig, max_length: int) \
        -> tuple[str, nn.Module]:
    return Text.modality_code(), nn.Sequential(
        SubclipTextExtract(interleaved=True, i_max_length=target_txt_config.i_max_length),
        MiniLMEmbedderTransform(),
        MultimediaPadding(max_length=math.ceil(max_length / target_txt_config.i_max_length))
    )
