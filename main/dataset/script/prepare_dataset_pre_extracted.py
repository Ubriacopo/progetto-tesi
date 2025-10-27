import dataclasses
import logging
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from main.core_data.media.audio import AudTargetConfig
from main.core_data.media.ecg import EcgTargetConfig
from main.core_data.media.ecg.config import EcgHydraConfig
from main.core_data.media.eeg import EegTargetConfig
from main.core_data.media.text import TxtTargetConfig
from main.core_data.media.video import VidTargetConfig
from main.dataset.amigos import loader as amigos_loader
from main.dataset.amigos import preprocessing as amigo_preprocessing
from main.dataset.deap import loader as deap_loader
from main.dataset.deap import preprocessing as deap_preprocessing


@dataclasses.dataclass
class PrepareDatasetConfigPreExtracted:
    dataset_name: str
    base_path: str
    data_path: str
    extraction_data_folder: str

    processor_name: str
    data_point_loader_name: str

    output_max_length: int
    output_path: str
    use_subpaths: bool = True
    eeg_config: Optional[EegTargetConfig] = dataclasses.field(default_factory=EegTargetConfig)
    ecg_config: EcgHydraConfig = dataclasses.field(default_factory=EcgHydraConfig)
    aud_config: AudTargetConfig = dataclasses.field(default_factory=AudTargetConfig)
    vid_config: VidTargetConfig = dataclasses.field(default_factory=VidTargetConfig)
    txt_config: TxtTargetConfig = dataclasses.field(default_factory=TxtTargetConfig)


# Register the schema as the default config
cs = ConfigStore.instance()
cs.store(name="prepare_dataset_config_pre_extracted", node=PrepareDatasetConfigPreExtracted)


@hydra.main(config_path="config", config_name="prepare_dataset_config_pre_extracted")
def main(cfg: PrepareDatasetConfigPreExtracted):
    OmegaConf.to_container(cfg, resolve=True)  # raises if MISSING present
    cfg.dataset_name = cfg.dataset_name.lower()
    if cfg.use_subpaths:
        cfg.data_path = cfg.base_path + cfg.data_path
        cfg.output_path = cfg.base_path + cfg.output_path
        cfg.extraction_data_folder = cfg.base_path + cfg.extraction_data_folder
        cfg.eeg_config.model_weights_path = cfg.base_path + cfg.eeg_config.model_weights_path
        cfg.txt_config.extracted_base_path = cfg.base_path + cfg.txt_config.extracted_base_path
    else:
        logging.warning("No subpaths are used. Ensure that all links are absolute.")
    # todo utils for this
    # Select the right domain for scripts
    preprocessing, loader = None, None
    if cfg.dataset_name == "amigos":
        preprocessing = amigo_preprocessing
        loader = amigos_loader
    elif cfg.dataset_name == "deap":
        preprocessing = deap_preprocessing
        loader = deap_loader

    pipe = getattr(preprocessing, cfg.processor_name)
    loader = getattr(loader, cfg.data_point_loader_name[0].upper() + cfg.data_point_loader_name[1:])
    processor = pipe(
        ecg_config=EcgTargetConfig.from_hydra(cfg.ecg_config), **{k: v for k, v in cfg.items() if k != "ecg_config"}
    )
    processor.run(loader(cfg.data_path))


if __name__ == "__main__":
    main()
