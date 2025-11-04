from hydra.utils import get_object

from main.core_data.data_point import FlexibleDatasetTransformWrapper
from main.core_data.media.assessment.default_transform_pipe import assessment_transform_pipe
from main.core_data.media.eeg.config import EegTargetConfig
from main.core_data.media.eeg.default_transform_pipe import eeg_transform_pipe
from main.core_data.media.metadata.metadata import Metadata
from main.core_data.media.metadata.transforms import MetadataToTensor
from main.core_data.media.video import VidTargetConfig
from main.core_data.media.video.default_transform_pipe import vid_vivit_interleaved_transform_pipe
from main.core_data.processing.preprocessing import TorchExportsSegmentsReadyPreprocessor
from main.dataset.deap.config import DeapConfig
from main.dataset.deap.loader import DeapPointsLoader
from main.dataset.utils import PreprocessingConfig
from main.utils.args import safe_call


@safe_call
def interleaved_preprocessor(output_path: str, extraction_data_folder: str, config: DeapConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "interleaved_preprocessor",
            vid_vivit_interleaved_transform_pipe(config),
            eeg_transform_pipe(config),
            # Audio and text do not exist so we don't use them.
            # TODO Check better if it was lost during processing.
            assessment_transform_pipe(),
            (Metadata.modality_code(), MetadataToTensor())
        ),
        extraction_data_folder=extraction_data_folder
    )


@safe_call
def vate_preprocessor(output_path: str, extraction_data_folder: str, config: DeapConfig):
    return TorchExportsSegmentsReadyPreprocessor(
        output_path=output_path,
        extraction_data_folder=extraction_data_folder,
        segment_pipeline=FlexibleDatasetTransformWrapper(
            "deap-vate-processor",
            vid_vivit_interleaved_transform_pipe(config),
            # Audio and text do not exist so we don't use them. TODO Check better if it was lost during processing.
            (Metadata.modality_code(), MetadataToTensor())
        )
    )


def preprocessing(config: PreprocessingConfig):
    # Build configuration from the one provided prior to call
    deap_config = DeapConfig(
        aud_target_config=config.aud_config,
        vid_target_config=config.vid_config,
        txt_target_config=config.txt_config,
        ecg_target_config=config.ecg_config,
        eeg_target_config=config.eeg_config,
        max_length=config.output_max_length
    )
    preprocessing_fn = get_object(config.preprocessing_function)
    # Either way I expect the same signature.
    preprocessor = preprocessing_fn(config.output_path, config.extraction_data_folder, deap_config)

    loader = DeapPointsLoader(base_path=config.base_path)
    preprocessor.run(loader=loader)
