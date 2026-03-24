from torch import nn

from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.assessment.assessment import Assessment, AssessmentLabels
from main.core_data.media.assessment.transform import AssessmentToTensor, RescaleAssessmentValue
from main.core_data.media.eeg.default_transform_pipe import eeg_transform_pipe, eeg_sample_pipeline
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.media.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe, \
    vid_vate_basic_transform_pipe
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor, \
    TorchExportsKdSegmentsReadyPreprocessor
from main.core_data.sampler import FixedIntervalsSegmenter
from main.dataset.deap.config import DeapConfig


def interleaved_preprocessor(output_path: str, extraction_data_folder: str, config: DeapConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "interleaved_preprocessor",
            vid_vivit_interleaved_transform_pipe(config),
            eeg_transform_pipe(config),
            # Audio and text do not exist so we cannot use them :(
            # assessment_transform_pipe(),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "deap-vate-sample-pipeline",
            eeg_sample_pipeline(config)
        ),
        extraction_data_folder=extraction_data_folder
    )


def interleaved_downstream_preprocessor(output_path: str, extraction_data_folder: str, config: DeapConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "segment_interleaved_downstream_preprocessor",
            eeg_transform_pipe(config),
            vid_vivit_interleaved_transform_pipe(config),
            # todo familairty rescaling to 1-9, also sort so tht it is like amigos
            (
                Assessment.modality_code(),
                nn.Sequential(
                    RescaleAssessmentValue(AssessmentLabels.FAMILIARITY, (1., 9.)),
                    AssessmentToTensor()
                ),

            ),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_downstream_preprocessor",
            eeg_sample_pipeline(config)
        ),
        segmenter=FixedIntervalsSegmenter(12)
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


def combined_preprocessor(output_path: str, extraction_data_folder: str, config: DeapConfig):
    return TorchExportsKdSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        student_segment_pipeline=FlexibleDatasetTransformWrapper(
            "DEAP-interleaved_preprocessor",
            vid_vivit_interleaved_transform_pipe(config),
            eeg_transform_pipe(config),
            # Audio and text do not exist so we cannot use them :(
            # assessment_transform_pipe(),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        student_sample_pipeline=FlexibleDatasetTransformWrapper(
            "DEAP-interleaved-sample-pipeline",
            eeg_sample_pipeline(config)
        ),
        teacher_segment_pipeline=FlexibleDatasetTransformWrapper(
            "DEAP-vate-processor",
            vid_vate_basic_transform_pipe(config),
            # Audio and text do not exist so we cannot use them :(
            (Metadata.modality_code(), MetadataToTensor())
        )
    )
