import os

import hydra
import lightning
import torch
import torchinfo
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import OmegaConf

from main.model.VATE.constrastive_model import MaskedContrastiveModel
from main.model.neegavi.factories.core import CoreFactory
from main.model.neegavi.helpers import default_trainer, build_easy_eegavi_module
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.neegavi.training import EasyEegAviKdVateMaskedModule
from main.model.script.hydra_beans import KdConfig
from main.utils.logging import make_logger


@hydra.main(config_path="../../../conf", config_name="train")
def main(cfg: KdConfig):
    lightning.seed_everything(cfg.seed, workers=True)
    logger = make_logger("hydra-main.train")
    logger.info(OmegaConf.to_yaml(cfg))

    print("ALLOC_CONF =", os.environ.get("PYTORCH_ALLOC_CONF"))
    print("CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    ckpt_path = "/home/jacopo/PycharmProjects/progetto-tesi/main/model/script/outputs/best/2026-03-23_12-18-47/checkpoints/last.ckpt"
    teacher = MaskedContrastiveModel(hidden_channels=cfg.teacher.hidden_channels, out_channels=cfg.teacher.out_channels)
    teacher.load_state_dict(torch.load(cfg.teacher_weights_path))
    teacher.eval()

    student = CoreFactory.default(**cfg.model.factory.args).build()
    module = EasyEegAviKdVateMaskedModule.load_from_checkpoint(
        ckpt_path,
        student=student,
        teacher=teacher,
        datamodule=KdTrainDataModule(
            dataset_paths=cfg.dataset_descriptors,
            batch_size=cfg.trainer.batch_size,
            seed=cfg.data_seed,
            dequantize_keys=["eeg", "aud", "vid", "txt", "ecg"],
            restore_iteration=31,
            train_fraction=1.0,
            # todo verify
            take_keys=[student.pivot.code] + student.fusion_keys()
        ),
        use_moco=True,
        kd_loss_weight=cfg.trainer.kd_loss_weight,
        fusion_loss_weight=cfg.trainer.fusion_loss_weight,
        lr=cfg.trainer.lr,
        seed=cfg.seed,
        strict=False
    )
    torchinfo.summary(module)

    module.datamodule.setup("")
    train_batches = module.datamodule.size("train")

    model_name = "EEGAVI-" + str(cfg.seed)
    profiling = False
    profiler = SimpleProfiler() if profiling else None
    logger.info(f"Working with seed={cfg.seed} and data_seed={cfg.data_seed}")
    trainer = default_trainer(epochs=40, model_name=model_name, profiler=profiler, limit_train_batches=train_batches,
                              # TODO mettere accumulation e mini batch size in config
                              monitor_key="val/fused/mrr_mean", accumulate_grad_batches=1)

    trainer.fit(module, datamodule=module.datamodule, ckpt_path=cfg.trainer.ckpt_path)
    if profiler is not None:
        logger.info(profiler.summary())

    logger.info("Finished training")


if __name__ == "__main__":
    main()
