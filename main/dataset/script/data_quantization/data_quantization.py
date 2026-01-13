import dataclasses

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_object
from omegaconf import OmegaConf

from main.dataset.base_config import DatasetConfig
from main.dataset.utils import PreprocessingConfig, DatasetUidStore
from main.utils.logging import make_logger


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
    """
    Since the data is erroneously stored in fp32 (should be fp16) we change it.
    By checking some of the data going from fp16 -> int8 + scales loss of info is almost 0.
    We so quantize to further reduce the space we use.

    For performance reasons I also should be changing the sharding.
    I want bigger shards. (Up to 3-5 GB each)
    """
    # allow extra keys only on txt_config
    loger = make_logger("prepare_ds_pre_extracted")
    loger.info(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.to_container(cfg, resolve=True)


if __name__ == "__main__":
    main()
