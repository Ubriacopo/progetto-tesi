import dataclasses

import hydra
import pandas as pd
from omegaconf import OmegaConf

from main.utils.logging import make_logger
import tensordict


@dataclasses.dataclass
class Config:
    student_dataset_path: str
    teacher_dataset_path: str
    uid_store_dataset_path: str


@hydra.main(version_base=None, config_name="config", config_path="config")
def main(cfg: Config):
    logger = make_logger("HydraSanitizeDatasets")
    # allow extra keys only on txt_config
    logger.info(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.to_container(cfg, resolve=True)

    # TODO: I dataset devono essere allineati all'index. Usa spec file
    a = pd.read_csv(cfg.student_dataset_path)

    for i in a.itertuples(index=False):
        td = tensordict.load(cfg.student_dataset_path + "/" + i.eid)
        td["meta"]["experiment"]

if __name__ == "__main__":
    main()
