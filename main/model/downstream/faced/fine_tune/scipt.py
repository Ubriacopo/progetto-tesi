import hydra
import lightning
import torch
from cbramod.models.cbramod import CBraMod
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from main.model.downstream.faced.config import SeedConfig
from main.model.downstream.faced.datamodule import FacedDataModule
from main.model.downstream.faced.model import DefaultFacedCBraFineTune
from main.model.downstream.faced.trainer import CBraModFacedClassificationTrainer
from main.utils.logging import make_logger

cs = ConfigStore.instance()
cs.store(name="train", node=SeedConfig)


@hydra.main(version_base=None, config_name="train")
def main(cfg: SeedConfig):
    lightning.seed_everything(cfg.seed, workers=True)
    logger = make_logger("hydra-main.train")
    logger.info(OmegaConf.to_yaml(cfg))

    datamodule = FacedDataModule(cfg.dataset_path, 1, batch_size=cfg.trainer_config.batch_size)

    if cfg.model_config.is_baseline:
        cbra_weights_path = "/home/jacopo/PycharmProjects/progetto-tesi/main/dependencies/cbramod/pretrained_weights.pth"
        backbone = CBraMod()
        backbone.load_state_dict(torch.load(cbra_weights_path, map_location="cpu"))
        model = DefaultFacedCBraFineTune(encoder=backbone, num_classes=cfg.labels)
        module = CBraModFacedClassificationTrainer(model, classes=cfg.labels, seed=cfg.seed)
    else:
        pass


if __name__ == "__main__":
    main()
