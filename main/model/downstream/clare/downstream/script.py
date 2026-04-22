import hydra
import lightning
import torch
from cbramod.models.cbramod import CBraMod
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from main.model.downstream.clare.datamodule import ClareDataModule
from main.model.downstream.clare.model import DefaultClareCBraFineTune, DefaultClareFineTune
from main.model.downstream.clare.trainer import ClaireClassificationTrainer, CBraModClareClassificationTrainer
from main.model.downstream.faced.config import SeedConfig
from main.model.downstream.utils import print_parameter_summary_by_module, print_trainable_parameters
from main.model.neegavi.config import CBraModEegModalityConfig
from main.model.neegavi.factories.fine_tune import FineTuneFactory
from main.model.neegavi.utils import get_model_ckpt_finetune
from main.utils.logging import make_logger

cs = ConfigStore.instance()
cs.store(name="train", node=SeedConfig)


@hydra.main(version_base=None, config_name="train")
def main(cfg: SeedConfig):
    lightning.seed_everything(cfg.seed, workers=True)
    logger = make_logger("hydra-main.train")
    logger.info(OmegaConf.to_yaml(cfg))

    cfg.dataset_path = "/home/jacopo/dataset/EEGAVI/DOWNSTREAM-CLARE/"
    cfg.labels = 2
    cfg.trainer_config.batch_size = 32

    datamodule = ClareDataModule(cfg.dataset_path, 20, batch_size=cfg.trainer_config.batch_size)
    cbra_weights_path = "/home/jacopo/PycharmProjects/progetto-tesi/main/dependencies/cbramod/pretrained_weights.pth"

    if cfg.model_config.is_baseline:
        backbone = CBraMod()
        backbone.load_state_dict(torch.load(cbra_weights_path, map_location="cpu"))
        model = DefaultClareCBraFineTune(encoder=backbone, num_classes=cfg.labels)
        module = CBraModClareClassificationTrainer(model, classes=cfg.labels, seed=cfg.seed)

        model_name = "CLARE-CBRA" + str(cfg.seed)
    else:
        backbone = FineTuneFactory.fine_tune_default(CBraModEegModalityConfig.default(cbra_weights_path)).build()
        backbone.load_state_dict(get_model_ckpt_finetune(weights_path=cfg.model_config.weights_path), strict=False)
        model = DefaultClareFineTune(encoder=backbone, num_classes=cfg.labels)
        module = ClaireClassificationTrainer(model, classes=cfg.labels, seed=cfg.seed)

        model_name = "CLARE" + str(cfg.seed)

    print_parameter_summary_by_module(model)
    print_trainable_parameters(model)
    monitor_key = "val_loss"

    trainer = lightning.Trainer(
        accelerator="gpu",
        devices=1,
        logger=TensorBoardLogger("tb_logs", name=model_name),
        callbacks=[
            RichProgressBar(),
            ModelCheckpoint(dirpath="checkpoints", filename=f"best-{cfg.seed}", every_n_epochs=1, save_top_k=1,
                            save_last=True, monitor=monitor_key, mode="min"),
            EarlyStopping(monitor=monitor_key, min_delta=0.0001, patience=5, mode="min", verbose=True),
        ],
        num_sanity_val_steps=0,
        precision="16-mixed",
        max_epochs=cfg.trainer_config.epochs,
        accumulate_grad_batches=2
    )

    trainer.fit(module, datamodule=datamodule)
    logger.info("Finished training")
    # Test now
    res = trainer.test(module, datamodule=datamodule, ckpt_path=f"checkpoints/best-{cfg.seed}.ckpt")
    logger.info(res)
    logger.info("Finished testing")


if __name__ == "__main__":
    main()
