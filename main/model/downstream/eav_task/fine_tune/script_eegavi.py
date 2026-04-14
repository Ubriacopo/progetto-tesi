import dataclasses

import hydra
import lightning
import torchinfo
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from main.model.downstream.core.probe_model import SimpleFineTuneProbe
from main.model.downstream.eav_task.datamodule import EavDataModule
from main.model.downstream.eav_task.fine_tune.trainer import EavClassificationTrainer
from main.model.neegavi.config import CBraModEegModalityConfig
from main.model.neegavi.factories.fine_tune import FineTuneFactory
from main.model.neegavi.utils import get_model_ckpt_finetune
from main.utils.logging import make_logger


@dataclasses.dataclass
class TrainerConfig:
    epochs: int = 50
    batch_size: int = 8


@dataclasses.dataclass
class ModelConfiguration:
    backbone_checkpoint: str = "/home/jacopo/PycharmProjects/progetto-tesi/main/model/script/outputs/best-2-attn-1-beta/2026-03-27_22-46-58/checkpoints/epochepoch=39-stepstep=102120.ckpt"


@dataclasses.dataclass
class SeedConfig:
    dataset_path: str = "/home/jacopo/dataset/EEGAVI/FINETUNE-DOWNSTREAM/interleaved-downstream-eav"
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
    cbra_weights_path = "/home/jacopo/PycharmProjects/progetto-tesi/main/dependencies/cbramod/pretrained_weights.pth"
    backbone = FineTuneFactory.fine_tune_default(CBraModEegModalityConfig.default(cbra_weights_path)).build()
    backbone.load_state_dict(get_model_ckpt_finetune(weights_path=cfg.model_config.backbone_checkpoint), strict=False)

    labels = 5
    model = SimpleFineTuneProbe(backbone=backbone, in_dim=384, out_dim=labels)
    module = EavClassificationTrainer(model, classes=labels, seed=cfg.seed)

    torchinfo.summary(module)
    monitor_key = "valid_loss"
    model_name = "EAV-CBRA" + str(cfg.seed)
    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger("tb_logs", name=model_name),
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(dirpath="checkpoints", filename=f"best-cbra-{cfg.seed}", every_n_epochs=1, save_top_k=1,
                            save_last=True, monitor=monitor_key, mode="min"),
            EarlyStopping(monitor=monitor_key, min_delta=0.0001, patience=5, mode="min", verbose=True),
        ],
        num_sanity_val_steps=0,
        precision="16-mixed",
        max_epochs=cfg.trainer_config.epochs,
    )

    trainer.fit(module, datamodule=datamodule)
    logger.info("Finished training")
    # Test now
    res = trainer.test(module, datamodule=datamodule, ckpt_path=f"checkpoints/best-cbra-{cfg.seed}.ckpt")
    logger.info(res)
    logger.info("Finished testing")


if __name__ == "__main__":
    main()
