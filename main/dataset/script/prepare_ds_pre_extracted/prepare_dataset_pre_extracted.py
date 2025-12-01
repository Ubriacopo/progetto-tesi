import dataclasses

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_object
from omegaconf import OmegaConf

from main.dataset.base_config import DatasetConfig
from main.dataset.utils import PreprocessingConfig, DatasetUidStore


@dataclasses.dataclass
class Config:
    preprocessing: PreprocessingConfig


cs = ConfigStore.instance()
OmegaConf.register_new_resolver("capitalize", lambda s: s.capitalize())
OmegaConf.register_new_resolver("uppercase", lambda s: s.upper())


@hydra.main(version_base=None, config_name="config", config_path="config")
def main(cfg: Config):
    # allow extra keys only on txt_config
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.to_container(cfg, resolve=True)

    config: DatasetConfig = get_object(cfg.preprocessing.config_classpath)(
        aud_target_config=cfg.preprocessing.aud_config,
        vid_target_config=cfg.preprocessing.vid_config,
        txt_target_config=cfg.preprocessing.txt_config,
        ecg_target_config=cfg.preprocessing.ecg_config,
        eeg_target_config=cfg.preprocessing.eeg_config,
        max_length=cfg.preprocessing.output_max_length
    )

    uid_store = DatasetUidStore(cfg.preprocessing.uid_store_path)
    # All preprocessing function have to adapt to the contract. TODO some sort of interface ensurance
    preprocessing_fn = get_object(cfg.preprocessing.preprocessing_pipeline)(
        cfg.preprocessing.output_path, cfg.preprocessing.extraction_data_folder, config
    )

    loader = get_object(cfg.preprocessing.loader_classpath)(
        base_path=cfg.preprocessing.data_path, dataset_uid_store=uid_store
    )
    preprocessing_fn.run(loader=loader, workers=4)


if __name__ == "__main__":
    main()
