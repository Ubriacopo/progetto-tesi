from torch import nn

from common.data.eeg import EEG
from common.data.eeg.config import EegTargetConfig
from common.data.eeg.transforms import EEGResample, EEGToTimePatches, CBraModEmbedderTransform
from common.data.signal.transforms import SubclipMneRaw, SignalToTensor


def eeg_transform_pipe(target_config: EegTargetConfig, source_fs: int) -> tuple[str, nn.Module]:
    return EEG.modality_code(), nn.Sequential(
        SubclipMneRaw(),
        EEGResample(tfreq=target_config.target_fs, sfreq=source_fs),
        SignalToTensor(),
        EEGToTimePatches(points_per_patch=target_config.target_fs, max_segments=target_config.max_segments),
        CBraModEmbedderTransform(weights_path=target_config.cbramod_weights_path)
    )
