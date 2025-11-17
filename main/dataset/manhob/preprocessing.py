from hydra.utils import get_object

from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.dataset.manhob.config import ManhobConfig
from main.dataset.manhob.loader import ManhobPointsLoader
from main.dataset.utils import PreprocessingConfig, DatasetUidStore


def interleaved_preprocessor(output_path: str, extraction_data_folder: str, config: ManhobConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "MANHOB-interleaved-processor",
        )
    )


def vate_preprocessor(output_path: str, extraction_data_folder: str, config: ManhobConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "MANHOB-vate-processor",
        )
    )
