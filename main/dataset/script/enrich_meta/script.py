import dataclasses

import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore
from hydra.utils import get_object
from omegaconf import OmegaConf

from dataset.base_config import DatasetConfig
from dataset.utils import DatasetUidStore
from main.utils.logging import make_logger
import tensordict

@dataclasses.dataclass
class TargetModelConfig:
    out_folder_name: str
    preprocessing_pipeline: str


@dataclasses.dataclass
class Config:
    dataset: PreprocessingConfig
    target_model: TargetModelConfig
    base_path: str


cs = ConfigStore.instance()
OmegaConf.register_new_resolver("capitalize", lambda s: s.capitalize())
OmegaConf.register_new_resolver("uppercase", lambda s: s.upper())


@hydra.main(version_base=None, config_name="base", config_path="config")
def main(cfg: Config):
    # allow extra keys only on txt_config
    loger = make_logger("prepare_ds_pre_extracted")
    loger.info(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.to_container(cfg, resolve=True)

    config: DatasetConfig = get_object(cfg.dataset.config_classpath)(
        aud_target_config=cfg.dataset.aud_config,
        vid_target_config=cfg.dataset.vid_config,
        txt_target_config=cfg.dataset.txt_config,
        ecg_target_config=cfg.dataset.ecg_config,
        eeg_target_config=cfg.dataset.eeg_config,
        max_length=cfg.dataset.output_max_length
    )

    uid_store = DatasetUidStore(cfg.dataset.uid_store_path)
    # All preprocessing function have to adapt to the contract. TODO some sort of interface ensurance
    preprocessing_fn = get_object(cfg.target_model.preprocessing_pipeline)(
        cfg.dataset.output_path, cfg.dataset.extraction_data_folder, config
    )

    loader = get_object(cfg.dataset.loader_classpath)(
        base_path=cfg.dataset.data_path, dataset_uid_store=uid_store
    )
    preprocessing_fn.run(loader=loader, workers=1)
    loger.info("Preprocessing finished.")


if __name__ == "__main__":
    main()
