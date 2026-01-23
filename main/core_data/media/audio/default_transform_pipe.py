import math

from torch import nn
from torchaudio.transforms import Resample
from torchvision.transforms import v2

from main.core_data.media.audio import AudTargetConfig
from main.core_data.media.audio import Audio
from main.core_data.media.audio.transforms import SubclipAudio, AudioToTensor, ToMono, MonoAudioSequencePartitioning, \
    WavLmEmbedderTransform, WavLmFeatureExtractorTransform, HubertBaseComputeFeature, HubertFeatureExtractor
from main.core_data.media.signal.transforms import SignalZeroMasking
from main.core_data.processing.transform import MultimediaPadding, ToSimpleMaskedObject, SequentialWithFallback, \
    DataQuantizationTransform, EmptyQuantizedObjectTransform
from main.dataset.base_config import DatasetConfig


def aud_wav2vec_interleaved_txt_extract_transform_pipe(config: DatasetConfig) -> tuple[str, nn.Module]:
    max_length = math.ceil(config.max_length / config.unit_seconds)
    return Audio.modality_code(), SequentialWithFallback(
        SubclipAudio(),
        AudioToTensor(),
        ToMono(),
        Resample(orig_freq=config.aud_source_config.fs, new_freq=config.aud_target_config.fs),
        MonoAudioSequencePartitioning(
            fs=config.aud_target_config.fs, sequence_duration_seconds=config.unit_seconds,
            resampler=SignalZeroMasking(max_length=config.unit_seconds, fs=config.aud_target_config.fs),
        ),
        WavLmFeatureExtractorTransform(sampling_rate=config.aud_target_config.fs),
        WavLmEmbedderTransform(map_to="cpu"),
        MultimediaPadding(max_length=math.ceil(config.max_length / config.unit_seconds)),
        DataQuantizationTransform(),
        default_remap=EmptyQuantizedObjectTransform(shape=(max_length, 199, 768), mask_shape=(max_length,)),
    )


def aud_vate_basic_transform_pipe(config: DatasetConfig) -> tuple[str, nn.Module]:
    return Audio.modality_code(), SequentialWithFallback(
        SubclipAudio(),  # In the split interval
        AudioToTensor(),
        ToMono(),
        HubertBaseComputeFeature(original_fs=config.aud_source_config.fs),
        HubertFeatureExtractor(),
        v2.Lambda(lambda x: x.to("cpu")),
        ToSimpleMaskedObject(stop_at_dim=-1),
        DataQuantizationTransform(),
        default_remap=EmptyQuantizedObjectTransform(shape=(768,), mask_shape=(1,), reduce_mask=True),
    )
