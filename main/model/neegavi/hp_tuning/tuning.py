import copy
import dataclasses
from abc import ABC

import lightning
import optuna
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import RichProgressBar

from main.app_config import AppConfig
from main.model.neegavi.helpers import build_easy_eegavi_module
from main.model.script.hydra_beans import KdConfig


@dataclasses.dataclass
class Definition[T](ABC):
    pass


@dataclasses.dataclass
class ChoiceDefinition[T](Definition[T]):
    values: list[T]


@dataclasses.dataclass
class StepDefinition[T](Definition[T]):
    low: T
    high: T
    step: T | None
    is_log: bool

    def __post_init__(self):
        # reject bool explicitly (since bool is a subclass of int)
        if isinstance(self.low, bool) or isinstance(self.high, bool) or isinstance(self.step, bool):
            raise TypeError("StepDefinition does not accept bool.")


@dataclasses.dataclass
class TuningSearchSpace:
    # CLIP-like contrastive: 5e-5 – 5e-4
    # Transformer encoders: 1e-5 – 3e-4
    # Thus: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    lr: Definition[float]
    # Only watch: [0.01 - 0.05]
    weight_decay: Definition[float]

    batch_size: Definition[int]
    attn_layers: Definition[int]  # Min-Max-Step
    beta: Definition[float]
    use_moco: Definition[bool] = ChoiceDefinition[bool]([True, False])

    def suggest(self, key: str, trial: optuna.Trial):
        if not hasattr(self, key):
            raise IndexError("Invalid trial key: {}".format(key))

        o = self.__getattribute__(key)
        if isinstance(o, ChoiceDefinition):
            return trial.suggest_categorical(key, o.values)

        if isinstance(o, StepDefinition):
            if isinstance(o.low, int):
                if o.step is None:
                    o.step = 1  # Default value deriving from optuna for ints
                return trial.suggest_int(key, low=o.low, high=o.high, step=o.step, log=o.is_log)
            return trial.suggest_float(key, low=o.low, high=o.high, step=o.step, log=o.is_log)

        raise TypeError("Invalid set object type: {}".format(type(o)))


def objective(trial: optuna.Trial, cfg: KdConfig, search_space: TuningSearchSpace) -> float:
    torch.manual_seed(AppConfig.SEED)  # Reproducibility
    # Tuned grid of parameters
    # Stage 1
    lr = search_space.suggest("lr", trial)
    batch_size = search_space.suggest("batch_size", trial)
    # Stage 2
    attn_layers = search_space.suggest("attn_layers", trial)
    use_moco = search_space.suggest("use_moco", trial)
    # alpha = trial.suggest_float(name="alpha(fusion)", low=0.01, high=1.0, step=0.1)  # Fixed at the moment
    beta = trial.suggest_float("beta(kd)", 1e-2, 10.0, log=True)

    custom_config = copy.deepcopy(cfg)
    # Stage 1: First to tune
    custom_config.trainer.lr = lr
    custom_config.trainer.batch_size = batch_size
    # Stage 2:
    custom_config.model.factory.args.attention_config = attn_layers
    custom_config.trainer.kd_loss_weight = beta
    custom_config.trainer.use_moco = use_moco
    # Stage 3: (If MoCo still on)
    # moco_queue_size

    # todo fai queste chiamate in una funziona sola visto che si duplica tra script
    module = build_easy_eegavi_module(custom_config)

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

    trainer.fit(module, datamodule=module.datamodule)
    results = trainer.validate(module, datamodule=module.datamodule)
    return results[0][monitor_key]
