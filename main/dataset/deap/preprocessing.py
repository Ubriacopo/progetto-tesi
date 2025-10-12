from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.assessment.default_transform_pipe import assessment_transform_pipe
from main.core_data.media.eeg.config import EegTargetConfig
from main.core_data.media.eeg.default_transform_pipe import eeg_transform_pipe
from main.core_data.media.video import VidTargetConfig
from main.core_data.media.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.dataset.deap.config import DeapConfig
from main.utils.args import safe_call


@safe_call
def deap_interleaved_preprocessor(
        output_max_length: int, output_path: str,
        eeg_config: EegTargetConfig,
        extraction_data_folder: str,
        vid_config: VidTargetConfig = VidTargetConfig(),
) -> TorchExportsSegmentsReadyPreprocessor:
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "interleaved_preprocessor",
            vid_vivit_interleaved_transform_pipe(vid_config, DeapConfig.Video.fps, output_max_length),
            eeg_transform_pipe(target_config=eeg_config, eeg_order=DeapConfig.EEG_CHANNELS,
                               source_fs=DeapConfig.EEG.fs, max_length=output_max_length),
            # Audio and text do not exist so we don't use them.
            # TODO Check better if it was lost during processing.
            assessment_transform_pipe()
        ),
        extraction_data_folder=extraction_data_folder
    )


@safe_call
def deap_vate_preprocessor(
        output_max_length: int, output_path: str,
        extraction_data_folder: str,
        vid_config: VidTargetConfig = VidTargetConfig(),
):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "deap-vate-processor",
            vid_vivit_interleaved_transform_pipe(vid_config, DeapConfig.Video.fps, output_max_length),
            # Audio and text do not exist so we don't use them. TODO Check better if it was lost during processing.
        )
    )
