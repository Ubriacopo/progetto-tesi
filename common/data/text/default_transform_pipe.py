import math

from torch import nn

from common.data.text import Text
from common.data.text.config import TxtTargetConfig
from common.data.text.transforms import WhisperClipTextExtract, SubclipTextExtract, TextRegistry, MiniLMEmbedderTransform
from common.data.transform import MultimediaPadding


def txt_from_aud_interleaved_txt_extract_transform_pipe(target_txt_config: TxtTargetConfig, max_length: int) \
        -> tuple[str, nn.Module]:
    return Text.modality_code(), nn.Sequential(
        WhisperClipTextExtract(device="cpu"),  # Extracts all texts
        TextRegistry(store_path=target_txt_config.registry_store_path),
        SubclipTextExtract(),
        MiniLMEmbedderTransform(),
        MultimediaPadding(max_length=math.ceil(max_length / target_txt_config.i_max_length))
    )
