from __future__ import annotations

from torch import nn
from torchaudio.transforms import Resample
from transformers import VivitImageProcessor

from common.data.amigos.config import AmigosConfig
from common.data.amigos.loader import AmigosPointsLoader
from common.data.audio import Audio
from common.data.audio.transforms import SubclipAudio, ToMono, W2VBertFeatureExtractorTransform, AudioZeroMasking, \
    AudioToTensor
from common.data.data_point import AgnosticDatasetTransformWrapper
from common.data.eeg import EEG
from common.data.eeg.transforms import EEGMneAddAnnotation, EEGToMneRawFromChannels, EEGResample, EEGToTensor, \
    EEGToTimePatches
from common.data.preprocessing import SegmenterPreprocessor
from common.data.sampler import FixedIntervalsSegmenter
from common.data.video import Video
from common.data.video.transforms import UnbufferedResize, SubclipVideo, VideoToTensor, RegularFrameResampling, \
    ViVitImageProcessorTransform, ViVitFeatureExtractorTransform


class AmigosPreprocessorFactory:
    @staticmethod
    def run_default(input_path: str, output_path: str, max_length: int = 8):
        return AmigosPreprocessorFactory.default(output_path, max_length).run(AmigosPointsLoader(input_path))

    @staticmethod
    def default(output_path: str, max_length: int = 8) -> SegmenterPreprocessor:
        p = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
        p.do_rescale = True
        p.do_normalize = True
        p.do_resize = True

        # todo EmbeddingsGeneratingPreprocessor cosi gestisce logica di aggregazione?
        return SegmenterPreprocessor(
            output_path=output_path,
            ch_names=AmigosConfig.CH_NAMES,
            ch_types=AmigosConfig.CH_TYPES,
            segmenter=FixedIntervalsSegmenter(max_length),
            pipeline=AgnosticDatasetTransformWrapper(
                "preprocessing-default",
                (
                    Video.modality_code(),
                    nn.Sequential(
                        UnbufferedResize((260, 260)),
                        SubclipVideo(),
                        VideoToTensor(),
                        RegularFrameResampling(32, drop_mask=True),
                        ViVitImageProcessorTransform(),
                        ViVitFeatureExtractorTransform()
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
                    nn.Sequential(
                        SubclipAudio(),
                        AudioToTensor(),
                        ToMono(),
                        Resample(44000, 16000),
                        AudioZeroMasking(8, 16000),
                        W2VBertFeatureExtractorTransform(),
                    )
                )
            )
        )

    @staticmethod
    def for_VATE(output_path: str, max_length: int = 8) -> SegmenterPreprocessor:
        return SegmenterPreprocessor(
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
