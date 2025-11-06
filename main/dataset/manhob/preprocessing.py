from hydra.utils import get_object

from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.dataset.manhob.config import ManhobConfig
from main.dataset.manhob.loader import ManhobPointsLoader
from main.dataset.utils import PreprocessingConfig


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


def preprocessing(config: PreprocessingConfig):
    # Build configuration from the one provided prior to call
    amigos_config = ManhobConfig(
        aud_target_config=config.aud_config,
        vid_target_config=config.vid_config,
        txt_target_config=config.txt_config,
        ecg_target_config=config.ecg_config,
        eeg_target_config=config.eeg_config,
        max_length=config.output_max_length
    )

    preprocessing_fn = get_object(config.preprocessing_pipeline)
    # Either way I expect the same signature.
    preprocessor = preprocessing_fn(config.output_path, config.extraction_data_folder, amigos_config)

    loader = ManhobPointsLoader(base_path=config.base_path)
    preprocessor.run(loader=loader)
