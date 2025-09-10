from __future__ import annotations

from torch import nn

from common.data.amigos.config import AmigosConfig
from common.data.amigos.loader import AmigosPointsLoader
from common.data.audio.transforms import SubclipAudio
from common.data.data_point import EEGDatasetTransformWrapper, AgnosticDatasetTransformWrapper
from common.data.eeg.transforms import EEGMneAddAnnotation
from common.data.preprocessing import SegmenterPreprocessor
from common.data.sampler import Segmenter, FixedIntervalsSegmenter
from common.data.video.transforms import UnbufferedResize, SubclipVideo


# todo ma serve? Non definisce logica sua. come ihrentiacen potrei fare composition
#   Sarebbe factory
class AmigosPreprocessor(SegmenterPreprocessor):
    @staticmethod
    def run_default(input_path: str, output_path: str, max_length: int = 8):
        return AmigosPreprocessor.default(output_path, max_length).run(AmigosPointsLoader(input_path))

    @staticmethod
    def default(output_path: str, max_length: int = 8) -> AmigosPreprocessor:
        return AmigosPreprocessor(
            output_path=output_path,
            segmenter=FixedIntervalsSegmenter(max_length),
            pipeline=AgnosticDatasetTransformWrapper(
                "preprocessing-default",
                ("vid", nn.Sequential(UnbufferedResize((260, 260)), SubclipVideo(), )),
                ("eeg", nn.Sequential(EEGMneAddAnnotation(), )),
                ("aud", nn.Sequential(SubclipAudio()))
            )
        )

    def __init__(self, output_path: str, segmenter: Segmenter,
                 pipeline: AgnosticDatasetTransformWrapper = None):
        super().__init__(
            output_path, segmenter, AmigosConfig.CH_NAMES, AmigosConfig.CH_TYPES, pipeline
        )


if __name__ == "__main__":
    AmigosPreprocessor.run_default(
        "../../../resources/AMIGOS/",
        "../../../resources/AMIGOS/processed/"
    )
