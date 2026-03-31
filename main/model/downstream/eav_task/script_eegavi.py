import dataclasses

import hydra
import lightning
import torchinfo
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from main.model.downstream.eav_task.datamodule import EavDataModule
from main.model.downstream.faced.trainer import FacedTrainer
from main.model.downstream.probe_model import SimpleLinearProbe
from main.model.neegavi.factory import Factory
from main.utils.logging import make_logger


@dataclasses.dataclass
class TrainerConfig:
    epochs: int = 50
    batch_size: int = 16


@dataclasses.dataclass
class ModelConfiguration:
    backbone_checkpoint: str = "/home/jacopo/PycharmProjects/progetto-tesi/main/model/script/outputs/best-2-attn-1-beta/2026-03-27_22-46-58/checkpoints/epochepoch=39-stepstep=102120.ckpt"


@dataclasses.dataclass
class SeedConfig:
    dataset_path: str = "/home/jacopo/dataset/EEGAVI/FUSION-DOWNSTREAM/DOWNSTREAM/interleaved-downstream-eav"
    seed: int = 42

    model_config: ModelConfiguration = dataclasses.field(default_factory=ModelConfiguration)
    trainer_config: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)


cs = ConfigStore.instance()
cs.store(name="train", node=SeedConfig)


@hydra.main(version_base=None, config_name="train")
def main(cfg: SeedConfig):
    lightning.seed_everything(cfg.seed, workers=True)
    logger = make_logger("hydra-main.train")
    logger.info(OmegaConf.to_yaml(cfg))

    datamodule = EavDataModule(cfg.dataset_path, 1, batch_size=cfg.trainer_config.batch_size)
    backbone = Factory.best_inference_loaded(cfg.model_config.backbone_checkpoint)

    labels = 5
    model = SimpleLinearProbe(backbone=backbone, in_dim=384, out_dim=labels)
    module = FacedTrainer(model, labels=labels, seed=cfg.seed)

    torchinfo.summary(module)
    monitor_key = "valid_loss"
    model_name = "EAV-LIN" + str(cfg.seed)
    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger("tb_logs", name=model_name),
        callbacks=[
            RichProgressBar(),
            EarlyStopping(monitor=monitor_key, min_delta=0.0001, patience=10, mode="min", verbose=True),
        ],
        num_sanity_val_steps=0,
        precision="16-mixed",
        max_epochs=cfg.trainer_config.epochs,
    )

    trainer.fit(module, datamodule=datamodule)
    logger.info("Finished training")
    # Test now
    res = trainer.test(module, datamodule=datamodule)
    logger.info(res)
    logger.info("Finished testing")


if __name__ == "__main__":
    main()
