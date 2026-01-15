import math

from torch import nn

from main.core_data.media.text import Text
from main.core_data.media.text import TxtTargetConfig
from main.core_data.media.text.transforms import SubclipTextExtract, MiniLMEmbedderTransform, \
    RestoreTextExtract, BertEmbeddings
from main.core_data.processing.transform import MultimediaPadding, ToSimpleMaskedObject, SequentialWithFallback, \
    EmptyObjectTransform, EmptyQuantizedObjectTransform, DataQuantizationTransform
from main.dataset.base_config import DatasetConfig


def shared_txt_transform_pipe(text_config: TxtTargetConfig, txt_extract_base_path: str):
    return Text.modality_code(), nn.Sequential(
        RestoreTextExtract(base_path=txt_extract_base_path),  # Extracts all texts
    )


def txt_from_aud_interleaved_txt_extract_transform_pipe(config: DatasetConfig) \
        -> tuple[str, nn.Module]:
    max_length = math.ceil(config.max_length / config.unit_seconds)
    return Text.modality_code(), SequentialWithFallback(
        SubclipTextExtract(interleaved=True, i_max_length=int(config.unit_seconds)),
        MiniLMEmbedderTransform(),
        MultimediaPadding(max_length=max_length),
        DataQuantizationTransform(),
        default_remap=EmptyQuantizedObjectTransform(shape=(max_length, 384), mask_shape=(max_length,)),
    )


def txt_vate_basic_transform_pipe() -> tuple[str, nn.Module]:
    # todo verify
    return Text.modality_code(), SequentialWithFallback(
        SubclipTextExtract(interleaved=False),
        BertEmbeddings(),
        ToSimpleMaskedObject(stop_at_dim=-1),
        DataQuantizationTransform(),
        default_remap=EmptyQuantizedObjectTransform(shape=(768,), mask_shape=(1,), reduce_mask=True),
    )
