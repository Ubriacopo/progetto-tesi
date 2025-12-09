import dataclasses

import hydra
import pandas as pd
from omegaconf import OmegaConf

from main.core_data.dataset import FlexibleEmbeddingsSpecMediaDataset
from main.core_data.media.eeg import EEG
from main.utils.logging import make_logger


@dataclasses.dataclass
class Config:
    student_dataset_path: str
    teacher_dataset_path: str


@hydra.main(version_base=None, config_name="config", config_path="config")
def main(cfg: Config):
    logger = make_logger("HydraSanitizeDatasets")
    # allow extra keys only on txt_config
    logger.info(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)
    OmegaConf.to_container(cfg, resolve=True)

    # TODO: I dataset devono essere allineati all'index. Usa spec file
    a = pd.read_csv(cfg.student_dataset_path).to_dict(orient="records")
    b = pd.read_csv(cfg.teacher_dataset_path).to_dict(orient="records")
    if len(a) == len(b):
        logger.info("Student dataset and teacher dataset have the same number of samples")
        logger.info("Procedure complete.")
        return

    remove_entries = []
    for entry_a in a:
        if not entry_a in b:
            remove_entries.append(entry_a)

    for entry_b in b:
        if not entry_b in a:
            remove_entries.append(entry_b)

    logger.debug("Removing entries: {}".format(remove_entries))
    logger.info(f"Removing {len(remove_entries)} entries")

if __name__ == "__main__":
    main()
