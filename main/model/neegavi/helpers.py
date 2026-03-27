import lightning
import optuna
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback

from main.model.VATE.constrastive_model import MaskedContrastiveModel
from main.model.neegavi.factory import Factory
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.neegavi.training import EasyEegAviKdVateMaskedModule
from main.model.script.hydra_beans import KdConfig


def build_easy_eegavi_module(cfg: KdConfig, train_data_frac: float = None) -> EasyEegAviKdVateMaskedModule:
    # Teacher has to be in evaluation mode (we don't need its gradients)
    teacher = MaskedContrastiveModel(hidden_channels=cfg.teacher.hidden_channels, out_channels=cfg.teacher.out_channels)
    teacher.load_state_dict(torch.load(cfg.teacher_weights_path))
    teacher.eval()

    student = Factory.default(**cfg.model.factory.args).build()

    return EasyEegAviKdVateMaskedModule(
        # Student model has args of default factory call in input from YAML
        student=student,
        teacher=teacher,
        attention_layers=cfg.model.factory.args["attention_config"],
        datamodule=KdTrainDataModule(
            dataset_paths=cfg.dataset_descriptors,
            batch_size=cfg.trainer.batch_size,
            seed=cfg.data_seed,
            dequantize_keys=["eeg", "aud", "vid", "txt", "ecg"],
            restore_iteration=None,
            train_fraction=train_data_frac,
            # todo verify
            take_keys=[student.pivot.code] + student.fusion_keys()
        ),
        batch_size=cfg.trainer.batch_size,
        use_kd=cfg.trainer.use_kd,
        use_moco=cfg.trainer.use_moco,
        kd_loss_weight=cfg.trainer.kd_loss_weight,
        fusion_loss_weight=cfg.trainer.fusion_loss_weight,
        lr=cfg.trainer.lr,
        seed=cfg.seed
    )


def default_trainer(epochs: int, model_name: str, profiler, limit_train_batches: int, monitor_key: str,
                    accumulate_grad_batches: int = 1, version: str = "0"):
    return lightning.Trainer(
        accelerator="gpu",
        devices=1,
        profiler=profiler,
        logger=TensorBoardLogger("tb_logs", name=model_name, version=version),
        callbacks=[
            RichProgressBar(),
            EarlyStopping(monitor=monitor_key, min_delta=0.002, patience=8, mode="max", verbose=True),
            ModelCheckpoint(
                dirpath="checkpoints",
                filename="epoch{epoch}-step{step}",
                every_n_train_steps=limit_train_batches,
                save_top_k=1,
                save_last=True,
                monitor=monitor_key,
                mode="max"
            )
        ],
        precision="16-mixed",
        max_epochs=epochs,
        max_steps=int(limit_train_batches * epochs),
        limit_train_batches=limit_train_batches,
        val_check_interval=1.0,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=limit_train_batches,
    )


def tuning_trainer(model_name: str, version: str, monitor_key: str, trial: optuna.Trial):
    return lightning.Trainer(
        accelerator="gpu", devices=1,
        logger=TensorBoardLogger("tb_logs", name=model_name, version=version),
        callbacks=[
            RichProgressBar(),
            PyTorchLightningPruningCallback(trial, monitor=monitor_key),
            EarlyStopping(monitor=monitor_key, min_delta=0.002, patience=10, mode="max", verbose=True)
        ],
    )
