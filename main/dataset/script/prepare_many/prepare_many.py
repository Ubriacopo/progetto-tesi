import dataclasses

import hydra
from hydra.utils import get_object
from omegaconf import OmegaConf

from main.dataset.base_config import DatasetConfig
from main.dataset.utils import DatasetUidStore, PreprocessingConfig
from main.utils.logging import make_logger


@dataclasses.dataclass
class TargetModelConfig:
    out_folder_name: str
    preprocessing_pipeline: str


@dataclasses.dataclass
class SingleConfig:
    dataset: PreprocessingConfig
    target_model: TargetModelConfig


@dataclasses.dataclass
class Config:
    datasets: list[SingleConfig]
    base_path: str

@hydra.main(version_base=None, config_name="multi", config_path="../prepare_ds_pre_extracted/config")
def main(cfg: Config):
    # allow extra keys only on txt_config
    loger = make_logger("prepare_ds_pre_extracted")
    loger.info(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.to_container(cfg, resolve=True)

    for ds_config in cfg.datasets:
        config: DatasetConfig = get_object(ds_config.dataset.config_classpath)(
            aud_target_config=ds_config.dataset.aud_config,
            vid_target_config=ds_config.dataset.vid_config,
            txt_target_config=ds_config.dataset.txt_config,
            ecg_target_config=ds_config.dataset.ecg_config,
            eeg_target_config=ds_config.dataset.eeg_config,
            max_length=ds_config.dataset.output_max_length
        )

        uid_store = DatasetUidStore(ds_config.dataset.uid_store_path)
        # All preprocessing function have to adapt to the contract. TODO some sort of interface ensurance
        preprocessing_fn = get_object(ds_config.target_model.preprocessing_pipeline)(
            ds_config.dataset.output_path, ds_config.dataset.extraction_data_folder, config
        )

        loader = get_object(ds_config.dataset.loader_classpath)(
            base_path=ds_config.dataset.data_path, dataset_uid_store=uid_store
        )
        preprocessing_fn.run(loader=loader, workers=1)
        loger.info("Preprocessing finished.")


if __name__ == "__main__":
    main()
