import hydra
import lightning
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

import hydra_utils
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.neegavi.training import EasyEegAviKdVateMaskedModule
from main.model.script.hydra_beans import KdConfig
from main.utils.logging import make_logger


@hydra.main(config_path="../../../conf", config_name="train")
def main(cfg: KdConfig):
    lightning.seed_everything(cfg.seed, workers=True)
    logger = make_logger("hydra-main.test")

    init_object = hydra_utils.init_trainlike_script(cfg)
    student = init_object.student
    teacher = init_object.teacher

    kd_train_datamodule = KdTrainDataModule(
        dataset_paths=cfg.dataset_descriptors,
        batch_size=cfg.trainer.batch_size,
        seed=cfg.data_seed,
        take_keys=[student.pivot.code] + student.fusion_keys()
    )

    module = EasyEegAviKdVateMaskedModule.load_from_checkpoint(
        cfg.checkpoint_path,
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

    trainer = L.Trainer(
        accelerator="gpu",
        logger=TensorBoardLogger("tb_logs", name="FINAL-TEST-NOFUSE" + str(cfg.seed)),
        devices=1,
        callbacks=[RichProgressBar()],
        # num_sanity_val_steps=1,
        precision="16-mixed",  # P6000 has no tensor cores
        log_every_n_steps=int(20),  # Plot every 1%
    )

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
