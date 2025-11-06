from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.dataset.manhob.config import ManhobConfig


def interleaved_preprocessor(output_path: str, extraction_data_folder: str, config: ManhobConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "manhob-interleaved-processor",
        )
    )


def vate_preprocessor(output_path: str, extraction_data_folder: str, config: ManhobConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "manhob-vate-processor",
        )
    )
