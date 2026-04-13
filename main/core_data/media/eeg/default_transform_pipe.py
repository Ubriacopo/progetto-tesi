from torch import nn

from main.core_data.media.eeg import EEG
from main.core_data.media.eeg.channel_canonical_order import EegOrder
from main.core_data.media.eeg.transforms import EEGResample, EEGToTimePatches, CBraModEmbedderTransform, EegTimePadding, \
    CanonicalOrderTransform
from main.core_data.media.signal.transforms import SubclipMneRaw, SignalToTensor, BandpassFilter
from main.core_data.processing.transform import DataQuantizationTransform
from main.dataset.base_config import DatasetConfig


def eeg_transform_pipe(config: DatasetConfig) \
        -> tuple[str, nn.Module]:
    if config.eeg_target_config is None or config.eeg_source_config is None:
        raise ValueError("EEG dataset config is required to work")

    return EEG.modality_code(), nn.Sequential(
        SubclipMneRaw(),
        EEGResample(tfreq=config.eeg_target_config.fs, sfreq=config.eeg_source_config.fs),
        SignalToTensor(),
        # Because we have fs=200 and CBraMod wants fs as points per patch max_segments=max_length
        EEGToTimePatches(points_per_patch=config.eeg_target_config.fs, max_segments=config.max_length),
        CanonicalOrderTransform(eeg_order=config.eeg_source_config.EEG_CHANNELS),
        CBraModEmbedderTransform(weights_path=config.eeg_target_config.model_weights_path),
        EegTimePadding(max_length=config.max_length),
        DataQuantizationTransform()
    )


def light_eeg_transform_pipe(config: DatasetConfig, eeg_order: EegOrder) -> tuple[str, nn.Module]:
    if config.eeg_target_config is None or config.eeg_source_config is None:
        raise ValueError("EEG dataset config is required to work")

    return EEG.modality_code(), nn.Sequential(
        SubclipMneRaw(),
        EEGResample(tfreq=config.eeg_target_config.fs, sfreq=config.eeg_source_config.fs),
        SignalToTensor(),
        EEGToTimePatches(points_per_patch=config.eeg_target_config.fs, max_segments=config.max_length),
        CanonicalOrderTransform(eeg_order=config.eeg_source_config.EEG_CHANNELS, canonical_order=eeg_order),
        # TODO padding + masking + quantization
        EegTimePadding(max_length=config.max_length),
        DataQuantizationTransform()
    )


def eeg_sample_pipeline(config: DatasetConfig) -> tuple[str, nn.Module]:
    return EEG.modality_code(), nn.Sequential(
        BandpassFilter(l_freq=0.5, h_freq=40.0, notch=50.0),
        EEGResample(tfreq=config.eeg_target_config.fs, sfreq=config.eeg_source_config.fs),
    )
