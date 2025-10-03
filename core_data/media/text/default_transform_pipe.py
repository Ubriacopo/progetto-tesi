import math

from torch import nn

from core_data.media.text import Text
from core_data.media.text import TxtTargetConfig
from core_data.media.text.transforms import SubclipTextExtract, MiniLMEmbedderTransform, \
    RestoreTextExtract, BertEmbeddings
from core_data.processing.transform import MultimediaPadding


def shared_txt_transform_pipe(text_config: TxtTargetConfig, txt_extract_base_path: str):
    return Text.modality_code(), nn.Sequential(
        RestoreTextExtract(base_path=txt_extract_base_path),  # Extracts all texts
    )


def txt_from_aud_interleaved_txt_extract_transform_pipe(target_txt_config: TxtTargetConfig, max_length: int) \
        -> tuple[str, nn.Module]:
    return Text.modality_code(), nn.Sequential(
        SubclipTextExtract(interleaved=True, i_max_length=target_txt_config.i_max_length),
        MiniLMEmbedderTransform(),
        MultimediaPadding(max_length=math.ceil(max_length / target_txt_config.i_max_length))
    )


def txt_vate_basic_transform_pipe() -> tuple[str, nn.Module]:
    return Text.modality_code(), nn.Sequential(
        SubclipTextExtract(interleaved=False),
        BertEmbeddings()
    )
