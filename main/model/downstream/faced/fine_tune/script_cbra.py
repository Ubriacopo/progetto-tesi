import hydra
import lightning
import torch
import torchinfo
from cbramod.models.cbramod import CBraMod
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from main.model.downstream.core.probe_model import SimpleCbraFineTune
from main.model.downstream.faced.config import SeedConfig
from main.model.downstream.faced.datamodule import FacedDataModule
from main.model.downstream.faced.trainer import CBraModFacedClassificationTrainer
from main.model.downstream.utils import print_parameter_summary_by_module, print_trainable_parameters
from main.utils.logging import make_logger

cs = ConfigStore.instance()
cs.store(name="train", node=SeedConfig)


@hydra.main(version_base=None, config_name="train")
def main(cfg: SeedConfig):
    lightning.seed_everything(cfg.seed, workers=True)
    logger = make_logger("hydra-main.train")
    logger.info(OmegaConf.to_yaml(cfg))

    datamodule = FacedDataModule(cfg.dataset_path, 1, batch_size=cfg.trainer_config.batch_size)
    cbra_weights_path = "/home/jacopo/PycharmProjects/progetto-tesi/main/dependencies/cbramod/pretrained_weights.pth"
    backbone = CBraMod()
    backbone.load_state_dict(torch.load(cbra_weights_path, map_location="cpu"))

    labels = 9
    model = SimpleCbraFineTune(encoder=backbone, in_dim=200, out_dim=labels)
    module = CBraModFacedClassificationTrainer(model, classes=labels, seed=cfg.seed)
    print_parameter_summary_by_module(model)
    print_trainable_parameters(model)

    torchinfo.summary(module)
    monitor_key = "val_loss"
    model_name = "FACED-CBRA" + str(cfg.seed)
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
        accumulate_grad_batches=4
    )

    trainer.fit(module, datamodule=datamodule)
    logger.info("Finished training")
    # Test now
    res = trainer.test(module, datamodule=datamodule, ckpt_path=f"checkpoints/best-cbra-{cfg.seed}.ckpt")
    logger.info(res)
    logger.info("Finished testing")


if __name__ == "__main__":
    main()
