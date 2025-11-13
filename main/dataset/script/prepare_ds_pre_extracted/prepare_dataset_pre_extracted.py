import dataclasses

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import get_object
from omegaconf import OmegaConf

from main.dataset.utils import PreprocessingConfig


@dataclasses.dataclass
class Config:
    preprocessing: PreprocessingConfig


cs = ConfigStore.instance()


@hydra.main(version_base=None, config_name="config", config_path="config")
def main(cfg: Config):
    # allow extra keys only on txt_config
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.to_container(cfg, resolve=True)
    fn = get_object(cfg.preprocessing.preprocessing_function)
    fn(cfg.preprocessing)


if __name__ == "__main__":
    main()
