from core_data.data_point import FlexibleDatasetTransformWrapper
from dataset.deap.config import DeapConfig
from core_data.media.eeg.config import EegTargetConfig
from core_data.media.eeg.default_transform_pipe import eeg_transform_pipe
from core_data.processing.preprocessing import TorchExportsSegmenterPreprocessor
from core_data.sampler import EegFeaturesAndRandLogUIntervalsSegmenter, Segmenter

from core_data.media.video import VidTargetConfig
from core_data.media.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe


def deap_interleaved_preprocessor(
        output_max_length: int, output_path: str,
        eeg_config: EegTargetConfig,
        segmenter: Segmenter = EegFeaturesAndRandLogUIntervalsSegmenter(
            min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
        ),
        vid_config: VidTargetConfig = VidTargetConfig(),
) -> TorchExportsSegmenterPreprocessor:
    return TorchExportsSegmenterPreprocessor(
        output_path=output_path,
        segmenter=segmenter,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "interleaved_preprocessor",
            vid_vivit_interleaved_transform_pipe(vid_config, DeapConfig.Video.fps, output_max_length),
            eeg_transform_pipe(target_config=eeg_config, eeg_order=DeapConfig.EEG_CHANNELS,
                               source_fs=DeapConfig.EEG.fs, max_length=output_max_length),
            # Audio and text do not exist so we don't use them.
            # TODO Check better if it was lost during processing.
        )
    )


def deap_vate_preprocessor(
        output_max_length: int, output_path: str,
        segmenter: Segmenter = EegFeaturesAndRandLogUIntervalsSegmenter(
            min_length=2, max_length=32, num_segments=20, anchor_identification_hop=0.125, extraction_jitter=0.1
        ),
        vid_config: VidTargetConfig = VidTargetConfig(),
):
    return TorchExportsSegmenterPreprocessor(
        output_path=output_path,
        segmenter=segmenter,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "deap-vate-processor",
            vid_vivit_interleaved_transform_pipe(vid_config, DeapConfig.Video.fps, output_max_length),
            # Audio and text do not exist so we don't use them. TODO Check better if it was lost during processing.
        )
    )
