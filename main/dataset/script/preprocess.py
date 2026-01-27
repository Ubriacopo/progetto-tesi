import dataclasses

import hydra
from hydra.utils import get_object
from omegaconf import OmegaConf

from core_data.processing.preprocessing import Preprocessor
from dataset.base_config import DatasetConfig
from dataset.script.configs import PreprocessingTargetConfig
from dataset.utils import PreprocessingConfig, DatasetUidStore
from utils.logging import make_logger


@dataclasses.dataclass
class Config:
    dataset: PreprocessingConfig
    preprocessing: PreprocessingTargetConfig


@hydra.main(version_base=None, config_name="preprocessing", config_path="conf")
def main(cfg: Config):
    logger = make_logger("preprocess")
    logger.info(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.to_container(cfg, resolve=True)

    # Intervals are already extracted we process and put altogether
    # Build the dataset configuration of the project

    ds_config: DatasetConfig = get_object(cfg.dataset.config_classpath)(
        aud_target_config=cfg.dataset.aud_config,
        vid_target_config=cfg.dataset.vid_config,
        txt_target_config=cfg.dataset.txt_config,
        ecg_target_config=cfg.dataset.ecg_config,
        eeg_target_config=cfg.dataset.eeg_config,
        max_length=cfg.dataset.output_max_length
    )

    uid_store = DatasetUidStore(cfg.dataset.uid_store_path)
    preprocessing_init = get_object(cfg.preprocessing.preprocessing_pipeline)
    preprocessing_fn: Preprocessor = preprocessing_init(
        cfg.dataset.output_path, cfg.dataset.extraction_data_folder, ds_config
    )

    loader_init = get_object(cfg.dataset.loader_classpath)
    loader = loader_init(base_path=cfg.dataset.data_path, dataset_uid_store=uid_store)

    preprocessing_fn.run(loader=loader)
    logger.info("Preprocessing finished.")


if __name__ == "__main__":
    main()
