# todo this next
import copy

import lightning
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import RichProgressBar
from sklearn.model_selection import learning_curve
from optuna.integration import PyTorchLightningPruningCallback
from main.app_config import AppConfig
from main.core_data.dataset import CachableDatasetDescriptor
from main.model.VATE.constrastive_model import MaskedContrastiveModel
from main.model.neegavi.config import EegModalityConfig, MaskedFeedForwardConfig, KdPerceiverModalityConfig
from main.model.neegavi.factory import Factory
from main.model.neegavi.model import EegInterAviModelConfiguration
from main.model.neegavi.train_utils import KdTrainDataModule
from main.model.neegavi.training import EasyEegAviKdVateMaskedModule


# todo cerca di tenere le cose il piu posisibli semplici
def objective(
        trial: optuna.Trial,

        teacher: MaskedContrastiveModel,
        dataset_descriptors: list[CachableDatasetDescriptor],

        eeg_config: EegModalityConfig,
        vid_config: KdPerceiverModalityConfig,
        aud_config: KdPerceiverModalityConfig,
        txt_config: KdPerceiverModalityConfig,
        ecg_config: MaskedFeedForwardConfig,
        custom_config: EegInterAviModelConfiguration,
        fusion_metric_codes: list[str],
        teacher_keys: list[str],
        seed: int,
        drop_p_min: float = 0.05,
        drop_p_max: float = 0.2,
        attention_max_layers: int = 4,
        attention_min_layers: int = 2,
):
    torch.manual_seed(AppConfig.SEED)  # Reproducibility
    # Tuned grid of parameters
    lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)  # TODO run on lr only first

    attn_layers = trial.suggest_int(name="attn_layers", low=attention_min_layers, high=attention_max_layers, step=1)
    drop_p = trial.suggest_float(name="drop_p", low=drop_p_min, high=drop_p_max, step=0.05)
    batch_size = trial.suggest_categorical(name="batch_size", choices=[32, 64, 128])
    use_moco = trial.suggest_categorical(name="use_moco", choices=[True, False])
    # alpha = trial.suggest_float(name="alpha(fusion)", low=0.01, high=1.0, step=0.1)  # Fixed at the moment
    beta = trial.suggest_float("beta(kd)", 1e-2, 10.0, log=True)

    custom_config = copy.deepcopy(custom_config)
    custom_config.drop_p = drop_p
    # todo fai queste chiamate in una funziona sola visto che si duplica tra script
    student = Factory.default(
        eeg_config=eeg_config,
        vid_config=vid_config,
        aud_config=aud_config,
        txt_config=txt_config,
        ecg_config=ecg_config,
        attention_config=attn_layers,  # Simple is strong, just choose how many to stack togheter
        custom_config=custom_config,
    )

    datamodule = KdTrainDataModule(
        dataset_paths=dataset_descriptors,
        batch_size=batch_size,
        dequantize_keys=["eeg", "aud", "vid", "txt", "ecg"],
        seed=AppConfig.SEED
    )

    module = EasyEegAviKdVateMaskedModule(
        student=student,
        teacher=teacher,
        datamodule=datamodule,
        use_moco=use_moco,
        kd_loss_weight=beta,
        fusion_loss_weight=1.0,
        lr=lr,
        seed=seed,
        fusion_metrics=fusion_metric_codes,
        kd_keys=list(map(lambda o: o.key, teacher_keys)),
    )

    max_epochs = 10
    monitor_key = "val_global/fused/bidirectional/mrr_mean"
    limit_train_batches = 2500  # 900 b=64 todo calculate from batch size
    trainer = lightning.Trainer(
        accelerator="gpu",
        logger=TensorBoardLogger("tb_logs", name=trial.study.study_name, version=str(trial.number)),
        devices=1,
        callbacks=[
            RichProgressBar(),
            PyTorchLightningPruningCallback(trial, monitor=monitor_key),
            EarlyStopping(monitor=monitor_key, min_delta=0.002, patience=10, mode="max", verbose=True)
        ],
        precision="16-mixed",  # P6000 has no tensor cores
        limit_train_batches=limit_train_batches,
        max_steps=8000,
        val_check_interval=500,  # validate every 500 train steps
        log_every_n_steps=int(20),  # Plot every 1%
        accumulate_grad_batches=4,  # This is to stabilize training todo pass from config
    )

    trainer.fit(module, datamodule=datamodule)
    results = trainer.validate(module, datamodule=datamodule)
    return results[0][monitor_key]
