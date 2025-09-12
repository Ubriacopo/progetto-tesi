from __future__ import annotations

from torch import nn
from torchaudio.transforms import Resample
from transformers import VivitImageProcessor
import common.data.video.transforms as vidtransforms
import common.data.audio.transforms as audtransforms

from common.data.amigos.config import AmigosConfig
from common.data.amigos.loader import AmigosPointsLoader
from common.data.audio import Audio
from common.data.audio.transforms import SubclipAudio, ToMono, W2VBertFeatureExtractorTransform, AudioZeroMasking, \
    AudioToTensor
from common.data.data_point import AgnosticDatasetTransformWrapper
from common.data.eeg import EEG
from common.data.eeg.transforms import EEGMneAddAnnotation, EEGToMneRawFromChannels, EEGResample, EEGToTensor, \
    EEGToTimePatches
from common.data.preprocessing import TorchExportsSegmenterPreprocessor
from common.data.sampler import FixedIntervalsSegmenter
from common.data.video import Video
from common.data.video.transforms import UnbufferedResize, SubclipVideo, VideoToTensor, RegularFrameResampling, \
    ViVitImageProcessorTransform
from common.model.embedding.predefined.vivit import ViViTFoundationEmbedder
from common.model.embedding.predefined.w2vbert import W2VBertFoundationEmbedder


def setup_vivit_image_processor() -> None:
    p = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    p.do_rescale, p.do_normalize, p.do_resize = True


class AmigosPreprocessorFactory:
    @staticmethod
    def run_default(input_path: str, output_path: str, max_length: int = 8):
        return AmigosPreprocessorFactory.default(output_path, max_length).run(AmigosPointsLoader(input_path))

    @staticmethod
    def interleaved(output_path: str, max_length: int = 8):
        setup_vivit_image_processor()

        vid_transform = nn.Sequential(
            # TODO Forse serve unbuffered resize per evitare di caricare troppa roba
            vidtransforms.VideoToTensor(),
            vidtransforms.ViVitImageProcessorTransform(),
            vidtransforms.SequenceResampling(
                original_fps=25, sequence_duration_seconds=2,
                frames_resampler=vidtransforms.RegularFrameResampling(32, drop_mask=True)
            ),
            ViViTFoundationEmbedder()
        )

        aud_transform = nn.Sequential(
            audtransforms.AudioSequenceResampler(
                original_fs=44100, sequence_duration_seconds=2,
                resampler=Resample(44000, 16000)),
            AudioZeroMasking(8, 16000),
            W2VBertFeatureExtractorTransform(force_time_seq=True),
            W2VBertFoundationEmbedder()
        )

        return TorchExportsSegmenterPreprocessor(
            output_path=output_path,
            ch_names=AmigosConfig.CH_NAMES,
            ch_types=AmigosConfig.CH_TYPES,
            # todo next is this
            segmenter=FixedIntervalsSegmenter(max_length),
            pipeline=AgnosticDatasetTransformWrapper(
                "interleaved",
                (Video.modality_code(), vid_transform),
                (Audio.modality_code(), aud_transform)
            )
        )

    @staticmethod
    def default(output_path: str, max_length: int = 8) -> TorchExportsSegmenterPreprocessor:
        p = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        p.do_rescale, p.do_normalize, p.do_resize = True

        # todo EmbeddingsGeneratingPreprocessor cosi gestisce logica di aggregazione?
        return TorchExportsSegmenterPreprocessor(
            output_path=output_path,
            ch_names=AmigosConfig.CH_NAMES,
            ch_types=AmigosConfig.CH_TYPES,
            segmenter=FixedIntervalsSegmenter(max_length),
            pipeline=AgnosticDatasetTransformWrapper(
                "preprocessing-default",
                (
                    Video.modality_code(),
                    nn.Sequential(
                        VideoToTensor(),
                        ViVitImageProcessorTransform(),
                        RegularFrameResampling(32, drop_mask=True),
                        ViViTFoundationEmbedder()
                    )
                ),
                (
                    EEG.modality_code(),
                    nn.Sequential(
                        EEGToMneRawFromChannels(AmigosConfig.CH_NAMES, AmigosConfig.CH_TYPES),
                        EEGMneAddAnnotation(),
                        EEGResample(200, 128),
                        EEGToTensor(),
                        EEGToTimePatches(200),
                    )
                ),
                (
                    Audio.modality_code(),
                    # todo continua qui
                    nn.Sequential(
                        SubclipAudio(),
                        AudioToTensor(),
                        ToMono(),
                        Resample(44000, 16000),
                        AudioZeroMasking(8, 16000),
                        # Lambda(lambda x: x.unsqueeze(0)),
                        W2VBertFeatureExtractorTransform(force_time_seq=True),
                        # tood unsqueeze non va prima
                        W2VBertFoundationEmbedder()
                    )
                )
            )
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


if __name__ == "__main__":
    AmigosPreprocessorFactory.run_default(
        "../../../resources/AMIGOS/",
        "../../../resources/AMIGOS/processed/"
    )
