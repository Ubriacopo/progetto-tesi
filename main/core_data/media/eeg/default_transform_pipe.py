from torch import nn

from main.core_data.media.eeg import EEG
from main.core_data.media.eeg.config import EegTargetConfig
from main.core_data.media.eeg.transforms import EEGResample, EEGToTimePatches, CBraModEmbedderTransform, EegTimePadding, \
    CanonicalOrderTransform
from main.core_data.media.signal.transforms import SubclipMneRaw, SignalToTensor
from main.dataset.base_config import DatasetConfig


def eeg_transform_pipe(config: DatasetConfig) \
        -> tuple[str, nn.Module]:
    return EEG.modality_code(), nn.Sequential(
        SubclipMneRaw(),
        EEGResample(tfreq=config.eeg_target_config.fs, sfreq=config.eeg_source_config.fs),
        SignalToTensor(),
        # Because we have fs=200 and CBraMod wants fs as points per patch max_segments=max_length
        EEGToTimePatches(points_per_patch=config.eeg_target_config.fs, max_segments=config.max_length),
        CanonicalOrderTransform(eeg_order=config.eeg_source_config.EEG_CHANNELS),
        CBraModEmbedderTransform(weights_path=config.eeg_target_config.model_weights_path),
        EegTimePadding(max_length=config.max_length),
    )
