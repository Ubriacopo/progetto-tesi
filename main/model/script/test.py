import hydra
import lightning
import lightning as L
import torchinfo
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

import hydra_utils
from main.model.neegavi.factories.core import CoreFactory
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.neegavi.training import EasyEegAviKdVateMaskedModule
from main.model.script.hydra_beans import KdConfig
from main.utils.logging import make_logger


@hydra.main(config_path="../../../conf", config_name="train")
def main(cfg: KdConfig):
    # cfg = OmegaConf.to_container(cfg, resolve=True)
    lightning.seed_everything(cfg.seed, workers=True)
    logger = make_logger("hydra-main.test")
    init_object = hydra_utils.init_trainlike_script(cfg)

    ckpt_path = "/home/jacopo/PycharmProjects/progetto-tesi/main/model/script/outputs/best-2-attn-1-beta/2026-03-27_22-46-58/checkpoints/epochepoch=39-stepstep=102120.ckpt"
    student = init_object.student
    teacher = init_object.teacher

    kd_train_datamodule = KdTrainDataModule(
        dataset_paths=cfg.dataset_descriptors,
        batch_size=cfg.trainer.batch_size,
        dequantize_keys=["eeg", "aud", "vid", "ecg"],
        seed=cfg.data_seed,
        take_keys=[student.pivot.code] + student.fusion_keys()
    )

    ckpt_path = "/home/jacopo/PycharmProjects/progetto-tesi/main/model/script/outputs/best-2-attn-1-beta/2026-03-27_22-46-58/checkpoints/epochepoch=39-stepstep=102120.ckpt"
    module = EasyEegAviKdVateMaskedModule.load_from_checkpoint(
        ckpt_path,
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
        strict=False
    )

    module.eval()

    for n, p in student.named_parameters():
        logger.info(n, p.requires_grad, p.grad is None)

    torchinfo.summary(module)
    limit_train_batches = 1  # 900 b=64

    trainer = L.Trainer(
        accelerator="gpu",
        logger=TensorBoardLogger("tb_logs", name="FINAL-TEST-" + str(cfg.seed)),
        devices=1,
        callbacks=[RichProgressBar()],
        # num_sanity_val_steps=1,
        precision="16-mixed",  # P6000 has no tensor cores
        log_every_n_steps=int(20),  # Plot every 1%
        limit_train_batches=limit_train_batches,
        # This experiment is considered in steps and not epochs because sampling is non-uniform and ds is hard to exhaust
        # without creating bias. Approaches like this are common and seen in CLIP/SigLIP-style applications
        # limit_train_batches=cfg.trainer.batches_per_epoch, Debug only
        val_check_interval=1.0,
        max_epochs=-1,  # or a very large number
    )

    if False:
        trainer.validate(module, datamodule=kd_train_datamodule)
    else:
        kd_train_datamodule.setup("test")
        c = kd_train_datamodule.test_for_ds()
        # Experiment on each ds independently
        for key, value in c.items():
            logger.info(f"Testing for dataset {key}")
            trainer.test(module, dataloaders=value)

        # Experiment on all also
        trainer.test(module, datamodule=kd_train_datamodule)


if __name__ == "__main__":
    main()
