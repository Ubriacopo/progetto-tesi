import math

from torch import nn

from main.core_data.media.text import Text
from main.core_data.media.text import TxtTargetConfig
from main.core_data.media.text.transforms import SubclipTextExtract, MiniLMEmbedderTransform, \
    RestoreTextExtract, BertEmbeddings
from main.core_data.processing.transform import MultimediaPadding, ToSimpleMaskedObject
from main.dataset.base_config import DatasetConfig


def shared_txt_transform_pipe(text_config: TxtTargetConfig, txt_extract_base_path: str):
    return Text.modality_code(), nn.Sequential(
        RestoreTextExtract(base_path=txt_extract_base_path),  # Extracts all texts
    )


def txt_from_aud_interleaved_txt_extract_transform_pipe(config: DatasetConfig) \
        -> tuple[str, nn.Module]:
    return Text.modality_code(), nn.Sequential(
        SubclipTextExtract(interleaved=True, i_max_length=int(config.unit_seconds)),
        MiniLMEmbedderTransform(),
        MultimediaPadding(max_length=math.ceil(config.max_length / config.unit_seconds))
    )


def txt_vate_basic_transform_pipe() -> tuple[str, nn.Module]:
    return Text.modality_code(), nn.Sequential(
        SubclipTextExtract(interleaved=False),
        BertEmbeddings(),
        ToSimpleMaskedObject(stop_at_dim=-1)
    )
