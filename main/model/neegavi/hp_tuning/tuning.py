from __future__ import annotations

import copy
import dataclasses
from abc import ABC

import lightning
import optuna
from lightning.pytorch.callbacks import EarlyStopping, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback


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
    lr: Definition[float]
    batch_size: Definition[int]
    attn_layers: Definition[int]
    beta: Definition[float]

    def suggest(self, key: str, trial: optuna.Trial):
        if not hasattr(self, key):
            raise IndexError("Invalid trial key: {}".format(key))

        o = self.__getattribute__(key)
        if isinstance(o, ChoiceDefinition):
            return trial.suggest_categorical(key, o.values)

        if isinstance(o, StepDefinition):
            if isinstance(o.low, int):
                step = 1 if o.step is None else o.step
                return trial.suggest_int(key, low=o.low, high=o.high, step=step, log=o.is_log)
            return trial.suggest_float(key, low=o.low, high=o.high, step=o.step, log=o.is_log)

        raise TypeError("Invalid set object type: {}".format(type(o)))

    @staticmethod
    def default() -> TuningSearchSpace:
        return TuningSearchSpace(
            # CLIP-like contrastive: 5e-5 – 5e-4
            # Transformer encoders: 1e-5 – 3e-4
            # Thus: [3e-5, 1e-4, 3e-4, 1e-3]
            lr=ChoiceDefinition(values=[3e-5, 1e-4, 3e-4, 1e-3]),
            batch_size=ChoiceDefinition(values=[32, 64, 128]),
            attn_layers=ChoiceDefinition(values=[2, 4, 6]),
            beta=ChoiceDefinition(values=[0.25, 0.5, 1, 2.0]),
        )

    @staticmethod
    def from_choices(
            lr: list = (3e-5, 1e-4, 3e-4, 1e-3),
            batch_size: list = (32, 64, 128),
            attn_layers: list = (2, 4, 6),
            beta: list = (0.25, 0.5, 1, 2.0)
    ):
        return TuningSearchSpace(
            lr=ChoiceDefinition(values=lr),
            batch_size=ChoiceDefinition(values=batch_size),
            attn_layers=ChoiceDefinition(values=attn_layers),
            beta=ChoiceDefinition(values=beta),
        )


def objective(trial: optuna.Trial, cfg: KdConfig, search_space: TuningSearchSpace, max_epochs: int = 5) -> float:
    """

    :param trial:
    :param cfg:
    :param search_space:
    :param max_epochs: Max number of training epochs (or steps?)
    :return:
    """
    lightning.seed_everything(cfg.seed, workers=True)
    # Tuned grid of parameters. We run multiple configs.
    custom_config = copy.deepcopy(cfg)
    custom_config.trainer.lr = search_space.suggest("lr", trial)
    custom_config.trainer.batch_size = search_space.suggest("batch_size", trial)
    custom_config.model.factory.args.attention_config = search_space.suggest("attn_layers", trial)
    custom_config.trainer.kd_loss_weight = search_space.suggest("beta", trial)

    module = build_easy_eegavi_module(custom_config)
    # monitor_key = "val_global/fused/bidirectional/mrr_mean"
    monitor_key = "val/fused/mrr_mean"
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
        max_steps=600,
        val_check_interval=500,  # validate every 500 train steps
        log_every_n_steps=20,  # Plot every 1%
        accumulate_grad_batches=1,  # This is to stabilize training todo pass from config
    )

    trainer.fit(module, datamodule=module.datamodule)

    results = trainer.validate(module, datamodule=module.datamodule)

    score = results[0].get(monitor_key)
    if score is None:
        raise KeyError(f"Metric '{monitor_key}' not found in validation results: {results[0].keys()}")

    return float(score)
