import math

from torch import nn

from main.core_data.media.ecg import EcgTargetConfig
from main.core_data.media.ecg.ecg import ECG
from main.core_data.media.ecg.transforms import EcgDataToTensor, EcgSequenceResampling, EcgFmEmbedderTransform
from main.core_data.media.signal.transforms import SubclipMneRaw, SignalZeroMasking
from main.core_data.processing.transform import MultimediaPadding, SequentialWithFallback, EmptyObjectTransform
from main.dataset.base_config import DatasetConfig


# todo rivisiona
def ecg_interleaved_transform_pipe(config: DatasetConfig) -> tuple[str, nn.Module]:
    max_length = math.ceil(config.max_length / config.unit_seconds)
    latent_size: int = 256
    patches: int = 32
    return ECG.modality_code(), SequentialWithFallback(
        SubclipMneRaw(),
        EcgDataToTensor(),
        EcgSequenceResampling(
            channels_first=True,
            sequence_duration_seconds=int(config.unit_seconds),
            resampler=SignalZeroMasking(max_length=config.unit_seconds, fs=config.ecg_target_config.fs),
        ),
        EcgFmEmbedderTransform(
            data_transform_fn=config.ecg_source_config.prepare_ecg, endpoint=config.ecg_target_config.fm_endpoint
        ),
        MultimediaPadding(max_length=max_length),
        default_remap=EmptyObjectTransform(shape=(max_length, patches, latent_size), mask_shape=(max_length,)),
    )
