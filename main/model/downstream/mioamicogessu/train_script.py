import dataclasses

import hydra
import lightning
import torchinfo
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import RichProgressBar, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import OmegaConf, MISSING

from main.model.downstream.mioamicogessu.data_utils import LinearProbeDataModule
from main.model.downstream.mioamicogessu.model import FusionLinearProbe
from main.model.downstream.mioamicogessu.training import FusionProbeTrainer
from main.model.neegavi.factory import Factory
from main.model.neegavi.utils import get_model_ckpt
from main.utils.logging import make_logger


@dataclasses.dataclass
class TrainerConfig:
    epochs: int = 10
    batch_size: int = 8


@dataclasses.dataclass
class FusionConfig:
    train_dataset: str = "/home/jacopo/dataset/EEGAVI/FUSION-DOWNSTREAM/DOWNSTREAM/interleaved-downstream"
    test_dataset: str = "/home/jacopo/dataset/EEGAVI/FUSION-DOWNSTREAM/DOWNSTREAM/interleaved-downstream-deap"

    seed: int = 1
    eegavi_ckpt: str = "/home/jacopo/PycharmProjects/progetto-tesi/main/model/script/outputs/best-2-attn-1-beta/2026-03-27_22-46-58/checkpoints/epochepoch=39-stepstep=102120.ckpt"

    trainer_config: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)

cs = ConfigStore.instance()
cs.store(name="train", node=FusionConfig)

@hydra.main(version_base=None,  config_name="train")
def main(cfg: FusionConfig):
    lightning.seed_everything(cfg.seed, workers=True)
    logger = make_logger("hydra-main.train")
    logger.info(OmegaConf.to_yaml(cfg))

    datamodule = LinearProbeDataModule(seed=cfg.seed, batch_size=cfg.trainer_config.batch_size)
    datamodule.add_dataset(cfg.train_dataset, 1, valid_fraction=0.1)
    datamodule.add_dataset(cfg.test_dataset, 1, test_fraction=1.0)
    datamodule.setup("train")

    # Load existing model
    # TODO rename factory + make method for this in factory
    ckpt = get_model_ckpt(weights_path=cfg.eegavi_ckpt)
    backbone = Factory.best_inference().build()
    # Load state of the seed ckpt
    backbone.load_state_dict(ckpt, strict=False)
    backbone.eval()

    # EEGAVI outputs a 384 embedding vector while FACED has 12 labels
    model = FusionLinearProbe(backbone=backbone, in_dim=384, out_dim=5)
    module = FusionProbeTrainer(model=model)
    torchinfo.summary(module)

    model_name = "EEGAVI-" + str(cfg.seed)
    profiling = False
    profiler = SimpleProfiler() if profiling else None
    monitor_key = "val_rmse"

    limit_train_batches = len(datamodule.train_dataset) // cfg.trainer_config.batch_size
    trainer = lightning.Trainer(
        profiler=profiler,
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger("tb_logs", name=model_name),
        callbacks=[
            RichProgressBar(),
            EarlyStopping(monitor=monitor_key, min_delta=0.002, patience=8, mode="min", verbose=True),
        ],
        num_sanity_val_steps=0,
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