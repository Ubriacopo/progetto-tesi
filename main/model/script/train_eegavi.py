import hydra
import lightning as L
import torch
import torchinfo
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler, SimpleProfiler
from pytorch_lightning.profilers import AdvancedProfiler

import hydra_utils
from main.app_config import AppConfig
from main.model.neegavi.train import EegAviKdVateMaskedSemiSupervisedModule
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.script.hydra_beans import KdConfig
from main.utils.logging import make_logger


@hydra.main(config_path="config", config_name="default")
def main(cfg: KdConfig):
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    logger = make_logger("hydra-main>train_eegavi")
    tb_logger = TensorBoardLogger("tb_logs", name="my_model")
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

        kd_loss_weight=cfg.trainer.kd_loss_weight,
        fusion_loss_weight=cfg.trainer.fusion_loss_weight,
        weakly_supervised_weight=cfg.trainer.weakly_supervised_weight,
        lr=cfg.trainer.lr,
        kd_temperature=cfg.trainer.kd_temperature,
        # All modalities contribute to fusion
        fusion_metrics=init_object.fusion_metric_codes,
        kd_keys=list(map(lambda o: o.key, init_object.teacher_keys)),
        dequantize_keys=["eeg", "aud", "vid", "txt", "ecg"]
    )

    for n, p in student.named_parameters():
        logger.info(n, p.requires_grad, p.grad is None)

    profiler = PyTorchProfiler(
        record_functions=True,  # important
        profile_memory=True,  # optional
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_prof"),
        with_stack=True,

        export_to_chrome=True,

        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA
        ],
    )

    profiler = SimpleProfiler()

    torchinfo.summary(module)
    trainer = L.Trainer(
        profiler=profiler,
        accelerator="gpu",
        logger=tb_logger,
        devices=1,
        max_epochs=cfg.trainer.epochs,
        callbacks=[
            # TQDMProgressBar(leave=True, refresh_rate=40)
        ],
        num_sanity_val_steps=0,
        precision="16-mixed",  # P6000 has no tensor cores
        log_every_n_steps=50,
        check_val_every_n_epoch=99,
        # This experiment is considered in steps and not epochs because sampling is non-uniform and ds is hard to exhaust
        # without creating bias. Approaches like this are common and seen in CLIP/SigLIP-style applications
        limit_train_batches=cfg.trainer.batches_per_epoch,
    )

    trainer.fit(module, datamodule=kd_train_datamodule)
    logger.info(profiler.summary())

    # trainer.test(module, datamodule=kd_train_datamodule)


if __name__ == "__main__":
    main()
