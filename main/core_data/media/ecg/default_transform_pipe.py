import math

from torch import nn

from main.core_data.media.ecg import EcgTargetConfig
from main.core_data.media.ecg.ecg import ECG
from main.core_data.media.ecg.transforms import EcgDataAsTensor, EcgSequenceResampling, EcgFmEmbedderTransform
from main.core_data.media.signal.transforms import SubclipMneRaw, SignalZeroMasking
from main.core_data.processing.transform import MultimediaPadding
from main.dataset.base_config import DatasetConfig


def ecg_interleaved_transform_pipe(config: DatasetConfig) -> tuple[str, nn.Module]:
    return ECG.modality_code(), nn.Sequential(
        SubclipMneRaw(),
        EcgDataAsTensor(),
        EcgSequenceResampling(
            original_fs=config.eeg_source_config.fs, channels_first=True,
            sequence_duration_seconds=int(config.unit_seconds),
            resampler=SignalZeroMasking(max_length=config.unit_seconds, fs=config.ecg_target_config.fs),
        ),
        EcgFmEmbedderTransform(
            data_transform_fn=config.ecg_source_config.prepare_ecg, endpoint=config.ecg_target_config.fm_endpoint
        ),
        MultimediaPadding(max_length=math.ceil(config.max_length / config.unit_seconds))
    )


def ecg_default_transform_pipe(target_config: EcgTargetConfig) -> tuple[str, nn.Module]:
    return ECG.modality_code(), nn.Sequential(
        SubclipMneRaw(),
        EcgDataAsTensor(),
        EcgFmEmbedderTransform(data_transform_fn=target_config.prepare, endpoint=target_config.fm_endpoint),
    )
