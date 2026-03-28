import dataclasses

import hydra
import lightning
import torchinfo
from lightning.pytorch.callbacks import RichProgressBar, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import OmegaConf

from main.model.downstream.linear_probe import SimpleLinearProbe
from main.model.downstream.linear_probe_datamodule import LinearProbeDataModule
from main.model.downstream.linear_probe_trainer import SimpleLinearProbeTrainer
from main.model.neegavi.factory import Factory
from main.utils.logging import make_logger


@dataclasses.dataclass
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 32


@dataclasses.dataclass
class SeedConfig:
    dataset_path: str
    seed: int = 42
    eegavi_ckpt: str = "/home/jfichera/PycharmProjects/progetto-tesi/main/model/script/outputs/best-seed-150/2026-03-23_12-31-47/checkpoints/epochepoch=17-stepstep=45954.ckpt"
    trainer_config: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)


@hydra.main(config_path="../../../../conf", config_name="train")
def main(cfg: SeedConfig):
    lightning.seed_everything(cfg.seed, workers=True)
    logger = make_logger("hydra-main.train")
    logger.info(OmegaConf.to_yaml(cfg))

    datamodule = LinearProbeDataModule(seed=cfg.seed, batch_size=cfg.trainer_config.batch_size)
    datamodule.add_dataset(cfg.dataset_path, test_fraction=0.15, valid_fraction=0.1)
    # Load existing model TODO: collate
    datamodule.set_train_collate_fn()
    backbone = Factory.best_inference_loaded(cfg.eegavi_ckpt)

    # EEGAVI outputs a 384 embedding vector while FACED has 12 labels
    module = SimpleLinearProbeTrainer(
        probe=SimpleLinearProbe(backbone=backbone, in_dim=384, out_dim=12)  # 12 dims of FACED
    )
    torchinfo.summary(module)

    model_name = "EEGAVI-" + str(cfg.seed)
    profiling = False
    profiler = SimpleProfiler() if profiling else None
    monitor_key = "val_rmse"
    limit_train_batches = datamodule.size("train")
    trainer = lightning.Trainer(
        profiler=profiler,
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger("tb_logs", name=model_name),
        callbacks=[
            RichProgressBar(),
            EarlyStopping(monitor=monitor_key, min_delta=0.002, patience=8, mode="min", verbose=True),
        ],
        precision="16-mixed",
        max_epochs=cfg.trainer_config.epochs,
        max_steps=int(limit_train_batches * cfg.trainer_config.epochs),
        limit_train_batches=limit_train_batches,
        val_check_interval=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=limit_train_batches,
    )

    trainer.fit(module, datamodule=datamodule)
    if profiler is not None:
        logger.info(profiler.summary())

    logger.info("Finished training")


if __name__ == "__main__":
    main()
