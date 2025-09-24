import math

from torch import nn

from common.data.ecg.config import EcgTargetConfig
from common.data.ecg.ecg import ECG
from common.data.ecg.transforms import EcgDataAsTensor, EcgSequenceResampling, EcgFmEmbedderTransform
from common.data.signal.transforms import SubclipMneRaw, SignalZeroMasking
from common.data.transform import MultimediaPadding


def ecg_interleaved_transform_pipe(target_config: EcgTargetConfig, original_fs: int, max_length: int) \
        -> tuple[str, nn.Module]:
    return ECG.modality_code(), nn.Sequential(
        SubclipMneRaw(),
        EcgDataAsTensor(),
        EcgSequenceResampling(
            # todo add masking ??
            original_fs=original_fs,
            sequence_duration_seconds=target_config.i_max_length,
            resampler=SignalZeroMasking(max_length=target_config.i_max_length, fs=target_config.target_fs),
            channels_first=True,
        ),
        EcgFmEmbedderTransform(data_transform_fn=target_config.prepare, endpoint=target_config.fm_endpoint),
        MultimediaPadding(max_length=math.ceil(max_length / target_config.i_max_length))
    )


def ecg_default_transform_pipe(target_config: EcgTargetConfig) -> tuple[str, nn.Module]:
    return ECG.modality_code(), nn.Sequential(
        SubclipMneRaw(),
        EcgDataAsTensor(),
        EcgFmEmbedderTransform(data_transform_fn=target_config.prepare, endpoint=target_config.fm_endpoint),
    )
