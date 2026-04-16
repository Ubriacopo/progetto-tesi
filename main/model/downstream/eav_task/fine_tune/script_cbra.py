import dataclasses

import hydra
import lightning
import torch
import torchinfo
from cbramod.models.cbramod import CBraMod
from hydra.core.config_store import ConfigStore
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from main.model.downstream.core.probe_model import SimpleFineTuneProbe, SimpleCbraFineTune
from main.model.downstream.eav_task.datamodule import EavDataModule
from main.model.downstream.eav_task.fine_tune.trainer import EavClassificationTrainer, CBraModEavClassificationTrainer
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


def print_parameter_summary_by_module(model):
    summary = {}

    for name, p in model.named_parameters():
        top = name.split(".")[0]
        if top not in summary:
            summary[top] = {"trainable": 0, "frozen": 0}

        if p.requires_grad:
            summary[top]["trainable"] += p.numel()
        else:
            summary[top]["frozen"] += p.numel()

    for module, counts in summary.items():
        total = counts["trainable"] + counts["frozen"]
        print(
            f"{module:20s} "
            f"trainable={counts['trainable']:>12,d} "
            f"frozen={counts['frozen']:>12,d} "
            f"total={total:>12,d}"
        )

def print_trainable_parameters(model):
    trainable = 0
    frozen = 0

    for name, p in model.named_parameters():
        n = p.numel()
        status = "TRAIN" if p.requires_grad else "FROZEN"
        print(f"{status:7s} | {n:>12,d} | {name}")
        if p.requires_grad:
            trainable += n
        else:
            frozen += n

    total = trainable + frozen
    pct = 100.0 * trainable / total if total > 0 else 0.0

    print("\nSummary")
    print(f"Trainable: {trainable:,}")
    print(f"Frozen:    {frozen:,}")
    print(f"Total:     {total:,}")
    print(f"% trainable: {pct:.2f}%")


@hydra.main(version_base=None, config_name="train")
def main(cfg: SeedConfig):
    lightning.seed_everything(cfg.seed, workers=True)
    logger = make_logger("hydra-main.train")
    logger.info(OmegaConf.to_yaml(cfg))

    datamodule = EavDataModule(cfg.dataset_path, 1, batch_size=cfg.trainer_config.batch_size)
    cbra_weights_path = "/home/jacopo/PycharmProjects/progetto-tesi/main/dependencies/cbramod/pretrained_weights.pth"
    backbone = CBraMod()
    backbone.load_state_dict(torch.load(cbra_weights_path, map_location="cpu"))

    labels = 5
    model = SimpleCbraFineTune(encoder=backbone, in_dim=200, out_dim=labels)
    module = CBraModEavClassificationTrainer(model, classes=labels, seed=cfg.seed)
    print_parameter_summary_by_module(model)
    print_trainable_parameters(model)

    torchinfo.summary(module)
    monitor_key = "val_loss"
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
