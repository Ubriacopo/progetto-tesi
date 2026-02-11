import hydra
import lightning as L
import torch
import torchinfo
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

import hydra_utils
from main.app_config import AppConfig
from main.model.neegavi.trainer.default import EegAviKdVateMaskedSemiSupervisedModule
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.script.hydra_beans import KdConfig
from main.utils.logging import make_logger


@hydra.main(config_path="config", config_name="default")
def main(cfg: KdConfig):
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = make_logger("hydra-main>train_eegavi")
    torch.manual_seed(AppConfig.SEED)  # Reproducibility
    init_object = hydra_utils.init_trainlike_script(cfg)

    student = init_object.student
    teacher = init_object.teacher

    kd_train_datamodule = KdTrainDataModule(
        dataset_paths=cfg.dataset_descriptors,
        batch_size=cfg.trainer.batch_size,
        batches_per_epoch=cfg.trainer.batches_per_epoch,
        seed=AppConfig.SEED
    )

    module = EegAviKdVateMaskedSemiSupervisedModule(
        student=student,
        teacher=teacher,

        datamodule=kd_train_datamodule,
        use_moco=True,
        kd_loss_weight=cfg.trainer.kd_loss_weight,
        fusion_loss_weight=cfg.trainer.fusion_loss_weight,
        lr=cfg.trainer.lr,
        kd_temperature=cfg.trainer.kd_temperature,
        # All modalities contribute to fusion
        fusion_metrics=init_object.fusion_metric_codes,
        kd_keys=list(map(lambda o: o.key, init_object.teacher_keys)),
        dequantize_keys=["eeg", "aud", "vid", "txt", "ecg"]
    )

    for n, p in student.named_parameters():
        logger.info(n, p.requires_grad, p.grad is None)

    profiler = SimpleProfiler()

    torchinfo.summary(module)
    m_key = "val/top1_mean"
    val_check_interval = 1000
    trainer = L.Trainer(
        # profiler=profiler,
        # enable_progress_bar=False,
        accelerator="gpu",
        logger=TensorBoardLogger("tb_logs", name="my_model"),
        devices=1,
        callbacks=[
            # TQDMProgressBar(leave=True, refresh_rate=40)
            EarlyStopping(monitor=m_key, min_delta=0.002, patience=8, mode="max", verbose=True),
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="step{step}",
                every_n_train_steps=val_check_interval,
                save_top_k=3,
                save_last=True,
                monitor=m_key,
                mode="max",
            ),
            RichProgressBar()
        ],
        # num_sanity_val_steps=1,
        precision="16-mixed",  # P6000 has no tensor cores
        log_every_n_steps=50,
        # This experiment is considered in steps and not epochs because sampling is non-uniform and ds is hard to exhaust
        # without creating bias. Approaches like this are common and seen in CLIP/SigLIP-style applications
        # limit_train_batches=cfg.trainer.batches_per_epoch, Debug only
        max_steps=1000000,  # 1000000
        val_check_interval=5000,
        max_epochs=-1,  # or a very large number
        accumulate_grad_batches=5, # This is to stabilize training
    )

    trainer.fit(module, datamodule=kd_train_datamodule)
    logger.info(profiler.summary())

    # trainer.test(module, datamodule=kd_train_datamodule)


if __name__ == "__main__":
    main()
