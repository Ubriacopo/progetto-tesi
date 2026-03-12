from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.ecg.default_transform_pipe import ecg_interleaved_transform_pipe
from main.core_data.media.eeg.default_transform_pipe import eeg_transform_pipe, eeg_sample_pipeline
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.dataset.dreamer.config import DreamerConfig


def interleaved_preprocessor(output_path: str, extraction_data_folder: str, config: DreamerConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "dreamer-interleaved-preprocessor",
            eeg_transform_pipe(config),
            ecg_interleaved_transform_pipe(config),
            # Audio and text do not exist so we cannot use them :(
            # assessment_transform_pipe(),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        sample_pipeline=FlexibleDatasetTransformWrapper(
            "reamer-vate-sample-pipeline",
            eeg_sample_pipeline(config)
        ),
        extraction_data_folder=extraction_data_folder
    )
