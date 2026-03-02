from random import seed

import hydra
import lightning as L
import torch
import torchinfo
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import SimpleProfiler

import hydra_utils
from main.app_config import AppConfig
from main.model.neegavi.training import EasyEegAviKdVateMaskedModule
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.script.hydra_beans import KdConfig
from main.utils.logging import make_logger


@hydra.main(config_path="../../../conf", config_name="train")
def main(cfg: KdConfig):
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = make_logger("hydra-main.train_eegavi")
    torch.manual_seed(AppConfig.SEED)  # Reproducibility
    init_object = hydra_utils.init_trainlike_script(cfg)

    student = init_object.student
    teacher = init_object.teacher

    kd_train_datamodule = KdTrainDataModule(
        dataset_paths=cfg.dataset_descriptors,
        batch_size=cfg.trainer.batch_size,
        restore_iteration=cfg.trainer.dl_start_index,  # To resume a training if necessary
        dequantize_keys=["eeg", "aud", "vid", "txt", "ecg"],
        seed=AppConfig.SEED
    )

    module = EasyEegAviKdVateMaskedModule(
        student=student,
        teacher=teacher,
        datamodule=kd_train_datamodule,
        use_moco=cfg.trainer.use_moco,
        kd_loss_weight=cfg.trainer.kd_loss_weight,
        fusion_loss_weight=cfg.trainer.fusion_loss_weight,
        lr=cfg.trainer.lr,
        seed=cfg.seed,
        # All modalities contribute to fusion
        fusion_metrics=init_object.fusion_metric_codes,
        kd_keys=list(map(lambda o: o.key, init_object.teacher_keys)),
    )

    for n, p in student.named_parameters():
        logger.info(n, p.requires_grad, p.grad is None)

    profiler = SimpleProfiler()

    torchinfo.summary(module)
    monitor_key = "val_global/fused/bidirectional/mrr_mean"

    model_name: str = (
        f"eegavi_{cfg.seed}"
        f"_{"moco-immediate" if cfg.trainer.use_moco else ""}"
        f"_b{cfg.trainer.batch_size}"
        f"_lr{cfg.trainer.lr}"
        "no-grad-stop"
    )
    limit_train_batches = 2500  # 900 b=64
    trainer = L.Trainer(
        # profiler=profiler,
        # enable_progress_bar=False,
        accelerator="gpu",
        logger=TensorBoardLogger("tb_logs", name=model_name, version="0"),
        devices=1,
        callbacks=[
            # TQDMProgressBar(leave=True, refresh_rate=40)
            EarlyStopping(
                monitor=monitor_key,
                min_delta=0.002,
                patience=20,
                mode="max",
                verbose=True
            ),
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="epoch{epoch}-step{step}",
                every_n_train_steps=limit_train_batches,
                save_top_k=1,
                save_last=True,
                monitor=monitor_key,
                mode="max",
            ),
            RichProgressBar()
        ],
        # num_sanity_val_steps=1,
        precision="16-mixed",  # P6000 has no tensor cores
        limit_train_batches=limit_train_batches,
        log_every_n_steps=int(20),  # Plot every 1%
        # This experiment is considered in steps and not epochs because sampling is non-uniform and ds is hard to exhaust
        # without creating bias. Approaches like this are common and seen in CLIP/SigLIP-style applications
        # limit_train_batches=cfg.trainer.batches_per_epoch, Debug only
        # max_steps=100_000,  # 1000000
        val_check_interval=1.0,
        max_epochs=1000,  # or a very large number
        accumulate_grad_batches=4,  # This is to stabilize training todo pass from config
    )
    # In case we want to restore a previous training we have to set ckpt_path
    trainer.fit(module, datamodule=kd_train_datamodule, ckpt_path=cfg.trainer.ckpt_path)
    logger.info(profiler.summary())

    # trainer.test(module, datamodule=kd_train_datamodule)


if __name__ == "__main__":
    main()
