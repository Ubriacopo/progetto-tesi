from torch import nn

from common.data.amigos.config import AmigosConfig
from common.data.audio import Audio
from common.data.data_point import AgnosticDatasetTransformWrapper
from common.data.deap.config import DeapConfig
from common.data.deap.loader import DeapPointsLoader
from common.data.eeg import EEG
from common.data.eeg.transforms import AddMneAnnotation
from common.data.preprocessing import TorchExportsSegmenterPreprocessor
from common.data.sampler import FeatureAndRandomLogUniformIntervalsSegmenter

from common.data.video import Video
from common.data.video.transforms import UnbufferedResize, SubclipVideo


# TODO Finish
def deap_interleaved_preprocessor(output_path: str,
                                  ecg_fm_endpoint: str = "localhost:7860/extract_features",
                                  cbramod_weights_path: str = "../../../dependencies/cbramod/pretrained_weights.pth",
                                  ) -> TorchExportsSegmenterPreprocessor:
    vid_gen = VidMultiSequenceTransformGenerator(
        max_length=30, original_fps=DeapConfig.Video.fps, sub_media_max_duration=4
    )

    eeg_gen = EEGTransformFlowGenerator(
        original_fs=DeapConfig.EEG.original_fs,
        target_fs=200,
        max_segments=10,
        cbra_weights_path=cbramod_weights_path
    )

    ecg_gen = EcgTransformFlowGenerator(
        ecg_fm_endpoint=ecg_fm_endpoint,
        original_fs=DeapConfig.EEG.original_fs,
        target_fs=128,
        sequence_duration_seconds=6,
        prepare_ecg=AmigosConfig.prepare_ecg
    )

    return TorchExportsSegmenterPreprocessor(
        output_path=output_path,
        ch_names=DeapConfig.CH_NAMES,
        ch_types=DeapConfig.CH_TYPES,
        segmenter=FeatureAndRandomLogUniformIntervalsSegmenter(
            min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
        ),
        pipeline=AgnosticDatasetTransformWrapper(
            "interleaved_preprocessor",
            vid_gen.transform(),
            eeg_gen.transform(),
            ecg_gen.transform(),
            # Audio and text do not exist so we don't use them.
        )
    )


class DeapPreprocessorFactory:
    @staticmethod
    def run_default(input_path: str, output_path: str, max_length: int = 8):
        return DeapPreprocessorFactory.default(output_path, max_length).run(DeapPointsLoader(input_path))

    @staticmethod
    def interleaved(output_path: str,
                    cbramod_weights_path: str = "../../../dependencies/cbramod/pretrained_weights.pth",
                    max_length: int = 30,
                    sub_media_max_length_seconds: int = 2,
                    ecg_endpoint: str = "localhost:7860/extract_features") -> TorchExportsSegmenterPreprocessor:
        vid_transform = nn.Sequential()
        return TorchExportsSegmenterPreprocessor(
            output_path=output_path,
            ch_names=DeapConfig.CH_NAMES,
            ch_types=DeapConfig.CH_TYPES,
            segmenter=FeatureAndRandomLogUniformIntervalsSegmenter(
                min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
            ),
            pipeline=AgnosticDatasetTransformWrapper(
                "interleaved",
                (Video.modality_code(), vid_transform)
            )
        )

    @staticmethod
    def default(output_path: str, max_length: int = 8):
        return TorchExportsSegmenterPreprocessor(
            output_path=output_path,
            ch_names=DeapConfig.CH_NAMES,
            ch_types=DeapConfig.CH_TYPES,
            segmenter=FeatureAndRandomLogUniformIntervalsSegmenter(
                min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
            ),
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
        )
