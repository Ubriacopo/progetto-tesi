from common.data.data_point import FlexibleDatasetTransformWrapper
from dataset.deap.config import DeapConfig
from common.data.eeg.config import EegTargetConfig
from common.data.eeg.default_transform_pipe import eeg_transform_pipe
from common.data.preprocessing import TorchExportsSegmenterPreprocessor
from common.data.sampler import EegFeaturesAndRandLogUIntervalsSegmenter, Segmenter

from common.data.video.config import VidTargetConfig
from common.data.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe, \
    vid_vivit_default_transform_pipe


def deap_interleaved_preprocessor(
        output_max_length: int, output_path: str,
        segmenter: Segmenter = EegFeaturesAndRandLogUIntervalsSegmenter(
            min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
        ),
        vid_config: VidTargetConfig = VidTargetConfig(),
        eeg_config: EegTargetConfig = EegTargetConfig("../../dependencies/cbramod/pretrained_weights.pth"),
) -> TorchExportsSegmenterPreprocessor:
    return TorchExportsSegmenterPreprocessor(
        output_path=output_path,
        ch_names=DeapConfig.CH_NAMES,
        ch_types=DeapConfig.CH_TYPES,
        segmenter=segmenter,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "interleaved_preprocessor",
            vid_vivit_interleaved_transform_pipe(vid_config, DeapConfig.Video.fps, output_max_length),
            eeg_transform_pipe(eeg_config, DeapConfig.EEG.fs, output_max_length),
            # Audio and text do not exist so we don't use them. TODO Check better if it was lost during processing.
        )
    )


def deap_default_preprocessor(
        output_max_length: int, output_path: str,
        segmenter: Segmenter = EegFeaturesAndRandLogUIntervalsSegmenter(
            min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
        ),
        vid_config: VidTargetConfig = VidTargetConfig(),
        eeg_config: EegTargetConfig = EegTargetConfig("../../dependencies/cbramod/pretrained_weights.pth"),
):
    return TorchExportsSegmenterPreprocessor(
        output_path=output_path,
        ch_names=DeapConfig.CH_NAMES,
        ch_types=DeapConfig.CH_TYPES,
        segmenter=segmenter,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "default_preprocessor",
            vid_vivit_default_transform_pipe(vid_config, DeapConfig.Video.fps, output_max_length),
            eeg_transform_pipe(eeg_config, DeapConfig.EEG.fs, output_max_length),
        ),
    )
