from __future__ import annotations

from torch import nn
from torchaudio.transforms import Resample
from torchvision.transforms import v2
from transformers import VivitImageProcessor

from common.data.amigos.config import AmigosConfig
from common.data.amigos.loader import AmigosPointsLoader
from common.data.audio import Audio
from common.data.audio.transforms import AudioToTensor
from common.data.audio.transforms.embedders import WavLmFeatureExtractorTransform, WavLmEmbedderTransform
from common.data.audio.transforms.transforms import SubclipAudio, ToMono, AudioSequenceResampler, AudioZeroMasking
from common.data.data_point import AgnosticDatasetTransformWrapper
from common.data.eeg import EEG
from common.data.eeg.transforms import AddMneAddAnnotationTransform, EEGToMneRaw, EEGResample, EEGToTensor, \
    EEGToTimePatches
from common.data.eeg.transforms.embedder import CBraModEmbedderTransform
from common.data.preprocessing import TorchExportsSegmenterPreprocessor
from common.data.sampler import FixedIntervalsSegmenter
from common.data.text import Text
from common.data.text.transforms import Wav2VecExtractFromAudio, MiniLMEmbedderTransform
from common.data.transform import MultimediaPadding, Parallel
from common.data.video import Video
from common.data.video.transforms import SubclipVideo, VideoToTensor, RegularFrameResampling, \
    ViVitImageProcessorTransform, VideoSequenceResampling
from common.model.embedding.predefined.vivit import ViViTFoundationEmbedder


def setup_vivit_image_processor() -> None:
    p = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    p.do_rescale, p.do_normalize, p.do_resize = True, True, True


class AmigosPreprocessorFactory:
    @staticmethod
    def run_default(input_path: str, output_path: str, max_length: int = 8):
        return AmigosPreprocessorFactory.default(output_path, max_length).run(AmigosPointsLoader(input_path))

    @staticmethod
    def interleaved(output_path: str, max_length: int = 8, sub_media_max_length_seconds: int = 2):
        setup_vivit_image_processor()

        target_fs = 200
        eeg_transform = nn.Sequential(
            EEGToMneRaw(AmigosConfig.CH_NAMES, AmigosConfig.CH_TYPES),
            AddMneAddAnnotationTransform(),
            EEGResample(target_fs, AmigosConfig.original_eeg_fs),
            EEGToTensor(),
            EEGToTimePatches(target_fs),
            # TODO verify
            CBraModEmbedderTransform()
        )

        vid_transform = nn.Sequential(
            SubclipVideo(),
            VideoToTensor(),
            ViVitImageProcessorTransform(),
            v2.Lambda(lambda x: x.pixel_values),
            VideoSequenceResampling(
                original_fps=AmigosConfig.original_vid_fps, sequence_duration_seconds=sub_media_max_length_seconds,
                frames_resampler=RegularFrameResampling(32, drop_mask=True)
            ),
            ViViTFoundationEmbedder(),
            MultimediaPadding(int(max_length / sub_media_max_length_seconds)),
        )

        target_audio_fs = 16000
        aud_transform = nn.Sequential(
            SubclipAudio(),
            AudioToTensor(),
            ToMono(),
            AudioSequenceResampler(
                original_fs=44100, sequence_duration_seconds=sub_media_max_length_seconds,
                resampler=nn.Sequential(
                    Resample(AmigosConfig.original_aud_fs, target_audio_fs),
                    AudioZeroMasking(sub_media_max_length_seconds, target_audio_fs, channels_first=False),
                )
            ),
            Parallel(
                nn.Sequential(
                    # TODO Custom audio cleaning is to do to see improvements?
                    Wav2VecExtractFromAudio(fs=target_audio_fs),  # Works a bit better.
                    # Speech2TextExtract(target_audio_fs  ),
                    MiniLMEmbedderTransform(),
                    MultimediaPadding(int(max_length / sub_media_max_length_seconds))
                ),
                nn.Sequential(
                    WavLmFeatureExtractorTransform(sampling_rate=target_audio_fs),
                    WavLmEmbedderTransform(),
                    MultimediaPadding(int(max_length / sub_media_max_length_seconds))
                ), as_dict=True, keys={Text.modality_code(), Audio.modality_code()},
            ),
        )

        return TorchExportsSegmenterPreprocessor(
            output_path=output_path,
            ch_names=AmigosConfig.CH_NAMES,
            ch_types=AmigosConfig.CH_TYPES,
            segmenter=FixedIntervalsSegmenter(max_length),
            pipeline=AgnosticDatasetTransformWrapper(
                "interleaved",
                (EEG.modality_code(), eeg_transform),
                (Video.modality_code(), vid_transform),
                (Audio.modality_code(), aud_transform),
                expand_nested=True, nested_keys=[Text.modality_code(), Audio.modality_code()],
            )

        )

    @staticmethod
    def default(output_path: str, max_length: int = 8) -> TorchExportsSegmenterPreprocessor:
        p = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        p.do_rescale, p.do_normalize, p.do_resize = True, True, True

        target_fs = 200
        eeg_transform = nn.Sequential(
            EEGToMneRaw(AmigosConfig.CH_NAMES, AmigosConfig.CH_TYPES),
            AddMneAddAnnotationTransform(),
            EEGResample(target_fs, AmigosConfig.original_eeg_fs),
            EEGToTensor(),
            EEGToTimePatches(target_fs),
        )
        max_vid_length = 32  # From ViVit
        vid_transform = nn.Sequential(
            SubclipVideo(),
            VideoToTensor(),
            ViVitImageProcessorTransform(),
            v2.Lambda(lambda x: x.pixel_values),
            # For ViVit masking makes no sense
            RegularFrameResampling(max_length=max_vid_length, drop_mask=True),
            ViViTFoundationEmbedder()
        )

        target_audio_fs = 16000
        aud_transform = nn.Sequential(
            SubclipAudio(),  # In the split interval
            AudioToTensor(),  # Transform to a tensor object
            ToMono(),  # Drop the dual channel audio and go to Mono
            Resample(AmigosConfig.original_aud_fs, target_audio_fs),
            AudioZeroMasking(max_length, target_audio_fs, channels_first=False),
            Parallel(
                nn.Sequential(
                    Wav2VecExtractFromAudio(fs=target_audio_fs),
                    MiniLMEmbedderTransform(),
                ),
                nn.Sequential(
                    WavLmFeatureExtractorTransform(sampling_rate=target_audio_fs),
                    WavLmEmbedderTransform(),
                ), as_dict=True, keys={Text.modality_code(), Audio.modality_code()},
            ),
        )

        txt_transform = nn.Sequential(

        )

        # todo EmbeddingsGeneratingPreprocessor cosi gestisce logica di aggregazione?
        return TorchExportsSegmenterPreprocessor(
            output_path=output_path,
            ch_names=AmigosConfig.CH_NAMES,
            ch_types=AmigosConfig.CH_TYPES,
            segmenter=FixedIntervalsSegmenter(max_length),
            pipeline=AgnosticDatasetTransformWrapper(
                "preprocessing-default",
                (Video.modality_code(), vid_transform),
                (EEG.modality_code(), eeg_transform),
                (Audio.modality_code(), aud_transform),
                (Text.modality_code(), txt_transform),
                expand_nested=True,
                nested_keys=[Text.modality_code(), Audio.modality_code()],
            ),
        )

    @staticmethod
    def for_VATE(output_path: str, max_length: int = 8) -> TorchExportsSegmenterPreprocessor:
        return TorchExportsSegmenterPreprocessor(
            output_path=output_path,
            ch_names=AmigosConfig.CH_NAMES,
            ch_types=AmigosConfig.CH_TYPES,
            segmenter=FixedIntervalsSegmenter(max_length),
            pipeline=AgnosticDatasetTransformWrapper(
                "preprocessing-VATE",
            ),
        )
