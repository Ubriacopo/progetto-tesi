import math

from torch import nn
from torchaudio.transforms import Resample
from torchvision.transforms import v2

from core_data.media.audio import AudTargetConfig
from core_data.media.audio import Audio
from core_data.media.audio.transforms import SubclipAudio, AudioToTensor, ToMono, AudioSequencePartitioning, \
    WavLmEmbedderTransform, WavLmFeatureExtractorTransform, HubertBaseComputeFeature, HubertFeatureExtractor
from core_data.media.signal.transforms import SignalZeroMasking
from core_data.processing.transform import MultimediaPadding


def aud_wav2vec_interleaved_txt_extract_transform_pipe(target_config: AudTargetConfig, fs: int, max_length: int) \
        -> tuple[str, nn.Module]:
    return Audio.modality_code(), nn.Sequential(
        SubclipAudio(),
        AudioToTensor(),
        ToMono(),
        Resample(orig_freq=fs, new_freq=target_config.fs),
        AudioSequencePartitioning(
            fs=target_config.fs, sequence_duration_seconds=target_config.i_max_length,
            resampler=SignalZeroMasking(max_length=target_config.i_max_length, fs=target_config.fs),
        ),
        WavLmFeatureExtractorTransform(sampling_rate=target_config.fs),
        WavLmEmbedderTransform(),
        MultimediaPadding(max_length=math.ceil(max_length / target_config.i_max_length))
    )


def aud_wav2vec_default_txt_extract_transform_pipe(target_config: AudTargetConfig, fs: int, max_length: int) \
        -> tuple[str, nn.Module]:
    return Audio.modality_code(), nn.Sequential(
        SubclipAudio(),  # In the split interval
        AudioToTensor(),  # Transform to a tensor object
        ToMono(),  # Drop the dual channel audio and go to Mono
        Resample(orig_freq=fs, new_freq=target_config.fs),
        SignalZeroMasking(max_length, target_config.fs, channels_first=False),
        WavLmFeatureExtractorTransform(sampling_rate=target_config.fs),
        WavLmEmbedderTransform()
    )


def aud_vate_basic_transform_pipe(fs: int) -> tuple[str, nn.Module]:
    return Audio.modality_code(), nn.Sequential(
        SubclipAudio(),  # In the split interval
        AudioToTensor(),
        ToMono(),
        HubertBaseComputeFeature(original_fs=fs),
        HubertFeatureExtractor(),
        v2.Lambda(lambda x: x.to("cpu"))
    )
