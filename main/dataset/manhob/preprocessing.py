from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.ecg.default_transform_pipe import ecg_interleaved_transform_pipe
from main.core_data.media.eeg.default_transform_pipe import eeg_transform_pipe
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.media.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.dataset.manhob.config import ManhobConfig


def interleaved_preprocessor(output_path: str, extraction_data_folder: str, config: ManhobConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "MANHOB-interleaved-processor",
            vid_vivit_interleaved_transform_pipe(config),
            eeg_transform_pipe(config),
            ecg_interleaved_transform_pipe(config),
            (Metadata.modality_code(), MetadataToTensor())
        )
    )


def vate_preprocessor(output_path: str, extraction_data_folder: str, config: ManhobConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "MANHOB-vate-processor",
            vid_vivit_interleaved_transform_pipe(config),
            (Metadata.modality_code(), MetadataToTensor())
        )
    )
