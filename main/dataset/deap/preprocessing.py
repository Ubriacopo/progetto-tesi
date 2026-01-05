from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.assessment.default_transform_pipe import assessment_transform_pipe
from main.core_data.media.eeg.default_transform_pipe import eeg_transform_pipe, eeg_sample_pipeline
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.media.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe, \
    vid_vate_basic_transform_pipe
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.dataset.deap.config import DeapConfig


def interleaved_preprocessor(output_path: str, extraction_data_folder: str, config: DeapConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "interleaved_preprocessor",
            vid_vivit_interleaved_transform_pipe(config),
            eeg_transform_pipe(config),
            # Audio and text do not exist so we cannot use them :(
            assessment_transform_pipe(),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "deap-vate-sample-pipeline",
            eeg_sample_pipeline(config)
        ),
        extraction_data_folder=extraction_data_folder
    )


def vate_preprocessor(output_path: str, extraction_data_folder: str, config: DeapConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "deap-vate-processor",
            vid_vate_basic_transform_pipe(config),
            # Audio and text do not exist so we cannot use them :(
            (Metadata.modality_code(), MetadataToTensor())
        ),
    )
