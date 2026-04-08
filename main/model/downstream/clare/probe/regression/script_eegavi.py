import dataclasses

import hydra
import lightning
import torchinfo
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from main.model.downstream.clare.datamodule import ClareDataModule
from main.model.downstream.clare.probe.regression.trainer import ClaireTrainer
from main.model.downstream.core.probe_model import Simple1DLinearProbe
from main.model.neegavi.factories.core import CoreFactory
from main.utils.logging import make_logger


@dataclasses.dataclass
class TrainerConfig:
    epochs: int = 50
    batch_size: int = 32


@dataclasses.dataclass
class ModelConfiguration:
    # Fixed
    backbone_checkpoint: str = "/home/jacopo/PycharmProjects/progetto-tesi/main/model/script/outputs/best-2-attn-1-beta/2026-03-27_22-46-58/checkpoints/epochepoch=39-stepstep=102120.ckpt"


@dataclasses.dataclass
class ClareConfig:
    dataset_path: str = "/home/jacopo/dataset/EEGAVI/FUSION-DOWNSTREAM/DOWNSTREAM/interleaved-downstream-clare"
    seed: int = 42

    model_config: ModelConfiguration = dataclasses.field(default_factory=ModelConfiguration)
    trainer_config: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)


cs = ConfigStore.instance()
cs.store(name="train", node=ClareConfig)


@hydra.main(version_base=None, config_name="train")
def main(cfg: ClareConfig):
    lightning.seed_everything(cfg.seed, workers=True)
    logger = make_logger("hydra-main.train")
    logger.info(OmegaConf.to_yaml(cfg))

    datamodule = ClareDataModule(cfg.dataset_path, 1, batch_size=cfg.trainer_config.batch_size)
    backbone = CoreFactory.best_inference_loaded(cfg.model_config.backbone_checkpoint)

    model = Simple1DLinearProbe(backbone=backbone, in_dim=384, out_dim=1)
    module = ClaireTrainer(model, seed=cfg.seed,lr=3e-4)

    torchinfo.summary(module)
    monitor_key = "val_loss"
    model_name = "CLARE-LIN-" + str(cfg.seed)
    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger("../tb_logs", name=model_name),
        callbacks=[
            RichProgressBar(),
            EarlyStopping(monitor=monitor_key, min_delta=0.0001, patience=15, mode="min", verbose=True),
            ModelCheckpoint(dirpath="checkpoints", filename=f"best-{cfg.seed}", every_n_epochs=1, save_top_k=1,
                            save_last=True, monitor=monitor_key, mode="min"),
        ],
        num_sanity_val_steps=0,
        precision="16-mixed",
        max_epochs=cfg.trainer_config.epochs,
    )

    trainer.fit(module, datamodule=datamodule)
    logger.info("Finished training")
    # Test now
    res = trainer.test(module, datamodule=datamodule, ckpt_path=f"checkpoints/best-{cfg.seed}.ckpt")
    logger.info(res)
    logger.info("Finished testing")


if __name__ == "__main__":
    main()
