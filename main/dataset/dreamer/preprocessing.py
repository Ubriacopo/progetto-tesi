from hydra.utils import get_object

from main.dataset.dreamer.config import DreamerConfig
from main.dataset.dreamer.loader import DreamerPointsLoader
from main.dataset.utils import PreprocessingConfig


def interleaved_preprocessor(output_path: str, extraction_data_folder: str, config: DreamerConfig):
    pass


def vate_preprocessor(output_path: str, extraction_data_folder: str, config: DreamerConfig):
    pass


def preprocessing(config: PreprocessingConfig):
    # Build configuration from the one provided prior to call
    amigos_config = DreamerConfig(
        aud_target_config=config.aud_config,
        vid_target_config=config.vid_config,
        txt_target_config=config.txt_config,
        ecg_target_config=config.ecg_config,
        eeg_target_config=config.eeg_config,
        max_length=config.output_max_length
    )

    preprocessing_fn = get_object(config.preprocessing_function)
    # Either way I expect the same signature.
    preprocessor = preprocessing_fn(config.output_path, config.extraction_data_folder, amigos_config)

    loader = DreamerPointsLoader(base_path=config.base_path)
    preprocessor.run(loader=loader)
