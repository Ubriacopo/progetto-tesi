from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.assessment.assessment import Assessment
from main.core_data.media.assessment.transform import AssessmentToTensor
from main.core_data.media.eeg.channel_canonical_order import EegCanonicalOrder
from main.core_data.media.eeg.default_transform_pipe import eeg_transform_pipe, eeg_sample_pipeline, \
    light_eeg_transform_pipe
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.core_data.sampler import FixedIntervalsSegmenter
from main.dataset.mahnob.config import MahnobConfig


def interleaved_preprocessor(output_path: str, extraction_data_folder: str, config: MahnobConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "faced-interleaved-processor",
            eeg_transform_pipe(config),
            (Assessment.modality_code(), AssessmentToTensor()),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "faced-sample-pipeline",
            eeg_sample_pipeline(config)
        ),
        segmenter=FixedIntervalsSegmenter(12)
    )




def interleaved_downstream_finetune_preprocessor(output_path: str, extraction_data_folder: str, config: EavConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "segment_interleaved_downstream_preprocessor",
            light_eeg_transform_pipe(config, EegCanonicalOrder()),
            (Assessment.modality_code(), AssessmentToTensor()),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "shared_interleaved_downstream_preprocessor",
            eeg_sample_pipeline(config)
        ),
        segmenter=FixedIntervalsSegmenter(12)
    )
