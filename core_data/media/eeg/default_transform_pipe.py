from torch import nn

from core_data.media.eeg import EEG
from core_data.media.eeg.config import EegTargetConfig
from core_data.media.eeg.transforms import EEGResample, EEGToTimePatches, CBraModEmbedderTransform, EegTimePadding
from core_data.media.signal.transforms import SubclipMneRaw, SignalToTensor


def eeg_transform_pipe(target_config: EegTargetConfig, source_fs: int, max_length: int) -> tuple[str, nn.Module]:
    return EEG.modality_code(), nn.Sequential(
        SubclipMneRaw(),
        EEGResample(tfreq=target_config.target_fs, sfreq=source_fs),
        SignalToTensor(),
        # Because we have fs=200 and CBraMod wants fs as points per patch max_segments=max_length
        EEGToTimePatches(points_per_patch=target_config.target_fs, max_segments=max_length),
        CBraModEmbedderTransform(weights_path=target_config.cbramod_weights_path),
        EegTimePadding(max_length=max_length),
    )
