import hydra
import lightning as L
import torch
import torchinfo
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.tuner import Tuner

from main.app_config import AppConfig
from main.model.neegavi.train import EegAviKdVateMaskedSemiSupervisedModule
from main.model.script import hydra_utils
from main.model.script.hydra_beans import KdConfig
from main.model.neegavi.train_utils import KdTrainDataModule
from main.utils.logging import make_logger


@hydra.main(config_path="config", config_name="default")
def main(cfg: KdConfig):
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = make_logger("hydra-main>train_eegavi")
    torch.manual_seed(AppConfig.SEED)  # Reproducibility
    init_object = hydra_utils.init_trainlike_script(cfg)

    student = init_object.student
    teacher = init_object.teacher

    module = EegAviKdVateMaskedSemiSupervisedModule(
        student=student,
        teacher=teacher,

        kd_loss_weight=cfg.trainer.kd_loss_weight,
        fusion_loss_weight=cfg.trainer.fusion_loss_weight,
        weakly_supervised_weight=cfg.trainer.weakly_supervised_weight,
        lr=cfg.trainer.lr,
        kd_temperature=cfg.trainer.kd_temperature,
        # All modalities contribute to fusion
        fusion_metrics=init_object.fusion_metric_codes,
        kd_keys=list(map(lambda o: o.key, init_object.teacher_keys)),
        batch_size=1
    )

    kd_train_datamodule = KdTrainDataModule(
        student_keys=init_object.student_keys,
        teacher_keys=init_object.teacher_keys,
        dataset_paths=list(zip(cfg.student_dataset_path, cfg.teacher_dataset_path)),
        student_pivot=cfg.model.pivot.code,  # Is for checks only could just remove it.
        batch_size=cfg.trainer.batch_size,
        batches_per_epoch=cfg.trainer.batches_per_epoch,
        seed=AppConfig.SEED
    )

    for n, p in student.named_parameters():
        logger.info(n, p.requires_grad, p.grad is None)

    torchinfo.summary(module)
    trainer = L.Trainer(
        profiler=SimpleProfiler(),
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.trainer.epochs,
        callbacks=[
            # TQDMProgressBar(leave=True, refresh_rate=40)
        ],
        num_sanity_val_steps=0,
        precision="16-mixed",  # P6000 has no tensor cores
        log_every_n_steps=50,
        # enable_progress_bar=False,
        # limit_train_batches=1
        check_val_every_n_epoch=0,
    )

    tuner = Tuner(trainer)
    tuner.scale_batch_size(module, mode="power")
    trainer.fit(module, kd_train_datamodule)
    logger.info(trainer.profiler.summary())


if __name__ == "__main__":
    main()
