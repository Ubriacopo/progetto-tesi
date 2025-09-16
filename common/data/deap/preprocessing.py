from torch import nn

from common.data.audio import Audio
from common.data.data_point import AgnosticDatasetTransformWrapper
from common.data.deap.config import DeapConfig
from common.data.deap.loader import DeapPointsLoader
from common.data.eeg import EEG
from common.data.eeg.transforms import AddMneAnnotation

from common.data.preprocessing import TorchExportsSegmenterPreprocessor
from common.data.sampler import FixedIntervalsSegmenter
from common.data.video import Video
from common.data.video.transforms import UnbufferedResize, SubclipVideo


class DeapPreprocessorFactory:
    @staticmethod
    def run_default(input_path: str, output_path: str, max_length: int = 8):
        return DeapPreprocessorFactory.default(output_path, max_length).run(DeapPointsLoader(input_path))

    @staticmethod
    def default(output_path: str, max_length: int = 8):
        return TorchExportsSegmenterPreprocessor(
            output_path=output_path,
            segmenter=FixedIntervalsSegmenter(max_length),
            pipeline=AgnosticDatasetTransformWrapper(
                "preprocessing-default",
                (
                    Video.modality_code(),
                    nn.Sequential(
                        UnbufferedResize((260, 260)),
                        SubclipVideo(),
                    )
                ),
                (
                    EEG.modality_code(),
                    nn.Sequential(
                        AddMneAnnotation()
                    ),
                ),
                (
                    Audio.modality_code(),
                    nn.Sequential()
                )
            ),
            ch_names=DeapConfig.CH_NAMES,
            ch_types=DeapConfig.CH_TYPES
        )
