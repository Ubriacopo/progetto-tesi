from common.data.audio.transforms import SubclipAudio
from common.data.data_point import EEGDatasetTransformWrapper
from common.data.deap.config import DeapConfig
from common.data.deap.loader import DeapPointsLoader
from common.data.eeg.transforms import EEGMneAddAnnotation
from common.data.preprocessing import EEGSegmenterPreprocessor
from common.data.sampler import FixedIntervalsSegmenter
from common.data.video.transforms import UnbufferedResize, SubclipVideo


class DeapPreprocessorFactory:
    @staticmethod
    def run_default(input_path: str, output_path: str, max_length: int = 8):
        return DeapPreprocessorFactory.default(output_path, max_length).run(DeapPointsLoader(input_path))

    @staticmethod
    def default(output_path: str, max_length: int = 8):
        return EEGSegmenterPreprocessor(
            output_path=output_path,
            segmenter=FixedIntervalsSegmenter(max_length),
            sample_pipeline=None,
            split_pipeline=EEGDatasetTransformWrapper(
                name="preprocessing-default",
                vid_transform=(
                    UnbufferedResize((260, 260)),
                    SubclipVideo(),
                ),
                eeg_transform=(
                    EEGMneAddAnnotation(),
                ),
                aud_transform=(
                    SubclipAudio(),
                )
            ),
            ch_names=DeapConfig.CH_NAMES,
            ch_types=DeapConfig.CH_TYPES
        )
