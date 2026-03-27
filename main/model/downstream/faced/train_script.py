import dataclasses
from pathlib import Path

import hydra
import lightning
import torchinfo
from lightning.pytorch.callbacks import RichProgressBar, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import OmegaConf

from main.core_data.ds.td_dataset import TdSegmentedExperimentDataset
from main.model.downstream.faced.data_utils import FacedProbeDataModule
from main.model.downstream.faced.model import FacedLinearProbe
from main.model.downstream.faced.training import FacedProbeTrainer
from main.model.neegavi.factory import Factory
from main.model.neegavi.helpers import default_trainer, build_easy_eegavi_module
from main.model.neegavi.utils import get_model_ckpt
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

    datamodule = FacedProbeDataModule(seed=cfg.seed, batch_size=cfg.trainer_config.batch_size)
    datamodule.add_dataset(cfg.dataset_path, test_fraction=0.15, valid_fraction=0.1)

    # Load existing model
    # TODO rename factory
    ckpt = get_model_ckpt(weights_path=cfg.eegavi_ckpt)
    backbone = Factory.best_inference().build()
    # Load state of the seed ckpt
    backbone.load_state_dict(ckpt, strict=False)
    backbone.eval()

    # EEGAVI outputs a 384 embedding vector while FACED has 12 labels
    model = FacedLinearProbe(backbone=backbone, in_dim=384, out_dim=12)
    module = FacedProbeTrainer(model=model)
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
