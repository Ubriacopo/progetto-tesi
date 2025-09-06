from __future__ import annotations

from torch import nn

from common.data.amigos.config import AmigosConfig
from common.data.amigos.loader import AmigosLoader
from common.data.audio.transforms import SubclipAudio
from common.data.data_point import EEGDatasetTransformWrapper
from common.data.eeg.transforms import EEGMneAddAnnotation
from common.data.preprocessing import EEGSegmenterPreprocessor
from common.data.sampler import Segmenter, FixedIntervalsSegmenter
from common.data.video.transforms import UnbufferedResize, SubclipVideo


class AmigosPreprocessor(EEGSegmenterPreprocessor):
    @staticmethod
    def execute_default(input_path: str, output_path: str, max_length: int = 8):
        return AmigosPreprocessor.default(output_path, max_length).run(AmigosLoader(input_path))

    @staticmethod
    def default(output_path: str, max_length: int = 8) -> AmigosPreprocessor:
        return AmigosPreprocessor(
            output_path=output_path,
            segmenter=FixedIntervalsSegmenter(max_length),
            sample_pipeline=None,
            split_pipeline=EEGDatasetTransformWrapper(
                name="preprocessing-default",
                vid_transform=nn.Sequential(
                    UnbufferedResize((260, 260)),
                    SubclipVideo(),
                ),
                eeg_transform=nn.Sequential(
                    EEGMneAddAnnotation(),
                ),
                aud_transform=nn.Sequential(
                    SubclipAudio()
                )
            )
        )

    def __init__(self, output_path: str, segmenter: Segmenter,
                 sample_pipeline: EEGDatasetTransformWrapper = None, split_pipeline: EEGDatasetTransformWrapper = None):
        super().__init__(
            output_path, segmenter, AmigosConfig.CH_NAMES, AmigosConfig.CH_TYPES, sample_pipeline, split_pipeline
        )


if __name__ == "__main__":
    AmigosPreprocessor.execute_default(
        "../../../resources/AMIGOS/",
        "../../../resources/AMIGOS/processed/"
    )
